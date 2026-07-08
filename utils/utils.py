import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from PIL import Image
import copy
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = '1'

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0



def test_single_volume(image, label, net, classes, patch_size=[224, 224], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    print(image.shape, label.shape)
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=1)
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()


    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

def vis_save(original_img, pred, save_path, idx, case):
    blue   = [30,144,255] # aorta
    green  = [0,255,0]    # gallbladder
    red    = [255,0,0]    # left kidney
    cyan   = [0,255,255]  # right kidney
    pink   = [255,0,255]  # liver
    yellow = [255,255,0]  # pancreas
    purple = [128,0,255]  # spleen
    orange = [255,128,0]  # stomach
    from matplotlib import pyplot as plt
    original_img = original_img * 255.0
    original_img = original_img.astype(np.uint8)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

    pred = zoom(pred, (224 / 512, 224 / 512), order=0)
    pred = pred.astype(np.uint8)
    pred = cv2.cvtColor(pred,cv2.COLOR_GRAY2BGR)

    original_img = np.where(pred==1, np.full_like(original_img, blue  ), original_img)
    original_img = np.where(pred==2, np.full_like(original_img, green ), original_img)
    original_img = np.where(pred==3, np.full_like(original_img, red   ), original_img)
    original_img = np.where(pred==4, np.full_like(original_img, cyan  ), original_img)
    original_img = np.where(pred==5, np.full_like(original_img, pink  ), original_img)
    original_img = np.where(pred==6, np.full_like(original_img, yellow), original_img)
    original_img = np.where(pred==7, np.full_like(original_img, purple), original_img)
    original_img = np.where(pred==8, np.full_like(original_img, orange), original_img)
    original_img = cv2.cvtColor(original_img,cv2.COLOR_BGR2RGB)
    path = save_path+f"/{case}_{str(idx)}.png"
    cv2.imwrite(path, original_img)
    print(f"{path} save over")

def plot_heat(img, mask, case, idx):
    img = (img * 255.0).astype(np.uint8)
    mask = mask.astype(np.uint8) * 255
    if idx % 25 == 0:
        from matplotlib import pyplot as plt
        plt.imshow(mask)
        plt.show()

    mask = mask.astype(np.uint8)
    if len(img.shape) != 3:
        img = np.expand_dims(img, 0).transpose(1,2,0)
        img = img.repeat(3, 2)

    img_w = img.shape[0] # 224 224
    img_h = img.shape[1]

    mask = cv2.resize(mask, (img_h, img_w), interpolation=cv2.INTER_CUBIC)


    heat_img = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)

    img_add = cv2.addWeighted(img, 0.3, heat_img, 0.7, 0)

    cv2.imwrite(f'./heatmap/{case}_{str(idx)}.png', img_add)  # 保存位置、保存名字
    print(f'case_{idx} saved!')

import numpy as np
import torch
import torch.nn as nn

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

class DiceScoreCoefficient(nn.Module):
    def __init__(self, n_classes):
        super(DiceScoreCoefficient, self).__init__()
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def fast_hist(self, label_true, label_pred, labels):
        mask = (label_true >= 0) & (label_true < labels)
        hist = np.bincount(labels * label_true[mask].astype(int) + label_pred[mask], minlength=labels ** 2,
        ).reshape(labels, labels)
        return hist

    def _dsc(self, mat):
        diag_all = np.sum(np.diag(mat))
        fp_all = mat.sum(axis=1)
        fn_all = mat.sum(axis=0)
        tp_tn = np.diag(mat)
        precision = np.zeros((self.n_classes)).astype(np.float32)
        recall = np.zeros((self.n_classes)).astype(np.float32)
        f2 = np.zeros((self.n_classes)).astype(np.float32)

        for i in range(self.n_classes):
            if (fp_all[i] != 0)and(fn_all[i] != 0):
                precision[i] = float(tp_tn[i]) / float(fp_all[i])
                recall[i] = float(tp_tn[i]) / float(fn_all[i])
                if (precision[i] != 0)and(recall[i] != 0):
                     f2[i] = (2.0*precision[i]*recall[i]) / (precision[i]+recall[i])
                else:
                    f2[i] = 0.0
            else:
                precision[i] = 0.0
                recall[i] = 0.0

        return f2


    ### main ###
    def forward(self, output, target):
        output = np.array(output)
        target = np.array(target)
        seg = np.argmax(output,axis=1)

        for lt, lp in zip(target, seg):
            self.confusion_matrix += self.fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

        dsc = self._dsc(self.confusion_matrix)

        return dsc
# coding: utf-8
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import datasets, transforms
import torchvision.models as models
import os
import argparse
import random
import sys
from tqdm import tqdm
import utils as ut
from mydataset import CVCClinicDB_Dataset
from model.launch_seg import launch_seg as UNet
from dsc import *



def Save_image(img, seg, ano, path):
    seg = np.argmax(seg, axis=1)
    img = img[0]
    img = np.transpose(img, (1, 2, 0))
    seg = seg[0]
    ano = ano[0]
    dst1 = np.zeros((seg.shape[0], seg.shape[1], 3))
    dst2 = np.zeros((seg.shape[0], seg.shape[1], 3))

    # class1 : background
    # class0 : polyp

    dst1[seg == 0] = [0.0, 0.0, 0.0]
    dst1[seg == 1] = [255.0, 255.0, 255.0]
    dst2[ano == 0] = [0.0, 0.0, 0.0]
    dst2[ano == 1] = [255.0, 255.0, 255.0]

    img = Image.fromarray(np.uint8(img * 255.0))
    dst1 = Image.fromarray(np.uint8(dst1))
    dst2 = Image.fromarray(np.uint8(dst2))

    img.save("{}_{}/Image/Inputs/{}.png".format(args.out, args.model, path), quality=95)
    dst1.save("{}_{}/Image/Seg/{}.png".format(args.out, args.model, path), quality=95)
    dst2.save("{}_{}/Image/Ano/{}.png".format(args.out, args.model, path), quality=95)

# training #
def train(epoch, iters):
    model.train()
    sum_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda(device)
        targets = targets.cuda(device)
        targets = targets.long()

        output = model(inputs)

        loss_ce = ce_loss(output, targets[:].long())
        loss_dice = dice_loss(output, targets, softmax=True)
        loss = 1.0* loss_ce + 0.5 * loss_dice

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()

        print("epoch:%d, iter %d / %d  train_Loss: %.4f, dice_loss:%.4f ce_loss:%.4f"
              % (epoch+1, iters + 1, args.maxiter, loss, loss_dice, loss_ce))
        with open(PATH_1, mode='a') as f:
            f.write("\t%d\t%f\n" % (iters + 1, loss))

        iters += 1

        adjust_learning_rate(optimizer, iters)

    return sum_loss / (batch_idx + 1), iters


# validation #
def test(epoch, model):
    model.eval()
    predict = []
    answer = []

    mIoU, mRec, mPre = [], [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda(device)
            targets = targets.cuda(device)
            targets = targets.long()

            output = model(inputs)

            output = F.softmax(output, dim=1)
            output = output.cpu().numpy()
            targets = targets.cpu().numpy()

            for j in range(args.batchsize):
                predict.append(output[j])
                answer.append(targets[j])
            IoU, Rec, Pre = ut.compute_metric(output, targets)
            mIoU.append(IoU)
            mRec.append(Rec)
            mPre.append(Pre)
            # Save_image(inputs, output, targets, batch_idx + 1)
        dsc = DiceScoreCoefficient(n_classes=args.classes)(predict, answer)
        mIoU = np.mean(mIoU)
        mRec = np.mean(mRec)
        mPre = np.mean(mPre)


    return dsc, mIoU, mRec, mPre


# adjust learning rate #
def adjust_learning_rate(optimizer, iters):
    lr = learning_rate * (1 - iters / args.maxiter) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




###### main ######
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--classes', '-c', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--batchsize', '-b', type=int, default=24)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_epochs', '-e', type=int, default=300)
    parser.add_argument('--maxiter', '-m', type=int, default=400000)
    parser.add_argument('--path', '-i', default='../../../data/cvc_clinicdb/')
    parser.add_argument('--out', '-o', type=str, default='result')
    parser.add_argument('--model', '-mo', type=str, default='mgcc')
    parser.add_argument('--gpu', '-g', type=str, default=0)
    parser.add_argument('--seed', '-s', type=int, default=42)
    args = parser.parse_args()
    gpu_flag = args.gpu

    dice_loss = DiceLoss(args.classes)
    ce_loss = nn.CrossEntropyLoss()

    # device #
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # save #
    if not os.path.exists("{}_{}".format(args.out, args.model)):
        os.mkdir("{}_{}".format(args.out, args.model))
    if not os.path.exists(os.path.join("{}_{}".format(args.out, args.model), "model")):
        os.mkdir(os.path.join("{}_{}".format(args.out, args.model), "model"))

    PATH_1 = "{}_{}/trainloss.txt".format(args.out, args.model)
    PATH_2 = "{}_{}/testloss.txt".format(args.out, args.model)
    PATH_3 = "{}_{}/DSC.txt".format(args.out, args.model)

    with open(PATH_1, mode='w') as f:
        pass
    with open(PATH_2, mode='w') as f:
        pass
    with open(PATH_3, mode='w') as f:
        pass

    # seed #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('high')

    # preprocceing #
    train_transform = ut.ExtCompose([ut.ExtResize((224, 224)),
                                     ut.ExtRandomRotation(degrees=90),
                                     ut.ExtRandomHorizontalFlip(),
                                     ut.ExtToTensor(),
                                     ])

    test_transform = ut.ExtCompose([ut.ExtResize((224, 224)),
                                   ut.ExtToTensor(),
                                   ])

    # data loader #
    data_train = CVCClinicDB_Dataset(root=args.path,
                                     dataset_type='train',
                                     transform=train_transform)
    data_test = CVCClinicDB_Dataset(root=args.path,
                                   dataset_type='test',
                                   transform=test_transform)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batchsize, shuffle=True, drop_last=False,
                                               num_workers=16)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batchsize, shuffle=False, drop_last=True,
                                             num_workers=16)

    seeds = [41,42,43]

    for seed in seeds:
        args.seed = seed
        # model #
        model = UNet(in_ch=3, start_ch=48, num_classes=args.classes, img_size=args.img_size,
                      granular_sizes=[args.img_size,args.img_size//2,args.img_size//4], up_factor=[1,2,4], loops=[1,2,3,4]).cuda(device)
        model.load_state_dict(torch.load('../pretrained/MGCC-pretrained.pth'), strict=False)

        print(len(train_loader))
        print(len(test_loader))

        # optimizer #
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

        ### training & validation ###
        sample = 0
        sample_loss = 10000000
        iters = 0

        iterator = tqdm(range(0, args.num_epochs), ncols=70)

        for epoch in iterator:
            args.maxiter = args.num_epochs * len(train_loader)
            loss_train, iters = train(epoch, iters)


        PATH_train = "{}_{}/model/{}_model_train.pth".format(args.out, args.model, args.seed)
        torch.save(model.state_dict(), PATH_train)
        dsc, mIoU, mRec, mPre = test(None, model)

        print("")
        print("seed=%d, Average DSC : %.4f, mean IoU: %.4f, mean Recall: %.4f, "
              "mean Precision:%.4f" % (args.seed, np.mean(dsc), mIoU, mRec, mPre))
        print("")


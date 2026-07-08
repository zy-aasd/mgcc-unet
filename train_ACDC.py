import logging
import random
from torch.cuda.amp import autocast, GradScaler

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
import matplotlib.pyplot as plt
from utils.utils import DiceLoss, test_single_volume
from torch.utils.data import DataLoader
from datasets.dataset_ACDC import ACDC_dataset, RandomGenerator
import argparse
from tqdm import tqdm
from model.launch_seg import launch_seg as seg_net
from torchvision import transforms
import numpy as np
from medpy.metric import dc, hd95
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = '1'


def val():
    logging.info("Validation ===>")
    dc_sum = 0
    model.eval()
    for i, val_sampled_batch in enumerate(valloader):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
        val_image_batch, val_label_batch = val_image_batch.type(torch.FloatTensor), val_label_batch.type(
            torch.FloatTensor)
        val_image_batch, val_label_batch = val_image_batch.cuda().unsqueeze(1), val_label_batch.cuda().unsqueeze(1)

        val_outputs = model(val_image_batch)
        val_outputs = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1).squeeze(0)
        dc_sum += dc(val_outputs.cpu().data.numpy(), val_label_batch[:].cpu().data.numpy())
    logging.info("avg_dsc: %f" % (dc_sum / len(valloader)))
    return dc_sum / len(valloader)


def inference(args, model, testloader, test_save_path=None):
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                          patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
            metric_list += np.array(metric_i)
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
                i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        metric_list = metric_list / len(testloader)
        for i in range(1, args.num_classes):
            logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        logging.info('seed %d Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (args.seed,performance, mean_hd95))
        logging.info("Testing Finished!")
        return performance, mean_hd95


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, help="batch size")
parser.add_argument("--lr", default=0.001, help="learning rate")
parser.add_argument("--max_epochs", default=120)
parser.add_argument("--img_size", default=224)
parser.add_argument("--save_path", default="../model/mgcc-acdc")
parser.add_argument("--n_gpu", default=1)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--list_dir", default="../../data/ACDC/lists_ACDC/")
parser.add_argument("--root_dir", default="../../data/ACDC/")
parser.add_argument("--volume_path", default="../../data/ACDC/")
parser.add_argument("--z_spacing", default=10)
parser.add_argument("--num_classes", default=4)
parser.add_argument('--test_save_dir', default='./predictions', help='saving predictions as nii!')
parser.add_argument("--n_skip", default=1)
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
args = parser.parse_args()


if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True

seeds = [1233,1234,1235]
for seed in seeds:
    args.seed = seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')


    model = seg_net(in_ch=3, start_ch=48, num_classes=args.num_classes, img_size=args.img_size,
            granular_sizes=[args.img_size, args.img_size // 2, args.img_size // 4], up_factor=[1, 2, 4],
            loops=[1, 2, 3, 4]).cuda()

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
    # train
    train_dataset = ACDC_dataset(args.root_dir, args.list_dir, split="train", transform=transforms.Compose(
        [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    Train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)
    # test
    db_test = ACDC_dataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    save_interval = args.n_skip  # int(max_epoch/6)

    iterator = tqdm(range(0, args.max_epochs), ncols=70)
    iter_num = 0

    Loss = []
    Test_Accuracy = []

    Best_dcs = 0.915
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s   %(levelname)s   %(message)s')

    max_iterations = args.max_epochs * len(Train_loader)
    base_lr = args.lr
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001, eps=1e-8, betas=(0.9, 0.999))
    model.load_state_dict(torch.load('pretrained/MGCC-pretrained.pth'), strict=False)

    scaler = GradScaler()
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    for epoch in iterator:
        model.train()
        train_loss = 0
        for i_batch, sampled_batch in enumerate(Train_loader):
            image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch[:], softmax=True)
            loss = loss_dice * 0.5 + loss_ce * 0.5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # lr_ = base_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
            train_loss += loss.item()

        Loss.append(train_loss / len(train_dataset))

        # if (epoch + 1) % save_interval == 0:
        #     avg_dcs = val()
        #
        #     if avg_dcs > Best_dcs:
        #         save_mode_path = os.path.join(args.save_path,
        #                                       'epoch={}_lr={}_avg_dcs={}.pth'.format(epoch, lr_, avg_dcs))
        #         # torch.save(model.state_dict(), save_mode_path)
        #         logging.info("save model to {}".format(save_mode_path))
        #         # temp = 1
        #
        #         dir = os.path.join(args.test_save_dir, 'epoch={}_lr={}_avg_dcs={}.pth'.format(epoch, lr_, avg_dcs))
        #         if not os.path.exists(dir):
        #             os.makedirs(dir)
        #         avg_dcs, avg_hd = inference(args, model, testloader, dir)
        #         Test_Accuracy.append(avg_dcs)

            # val visualization

            # fig2, ax2 = plt.subplots(figsize=(11, 8))
            # ax2.plot(range(int((epoch + 1) // save_interval)), Test_Accuracy)
            # ax2.set_title("Average val dataset dice score vs epochs")
            # ax2.set_xlabel("Epoch")
            # ax2.set_ylabel("Current dice score")
            # plt.savefig('val_dsc_vs_epochs_gauss.png')

            # plt.clf()
            # plt.close()

        if epoch >= args.max_epochs - 1:
            save_mode_path = os.path.join(args.save_path, 'epoch={}_lr={}.pth'.format(epoch, lr_))
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

            avg_dcs, avg_hd = inference(args, model, testloader, None)
            iterator.close()
            #break

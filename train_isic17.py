from datetime import datetime

import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from isic_loader import *
from model.launch_seg import launch_seg as seg_net

from isic_engine import *
import os
import sys
from configs.config_setting import setting_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1, 2, 3"
from isic17_utils import *


import warnings

warnings.filterwarnings("ignore")


def main(config, seed):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    config.seed = seed
    set_seed(config.seed)
    gpu_ids = [0]  # [0, 1, 2, 3]
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = isic_loader(path_Data=config.data_path, train=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)
    val_dataset = isic_loader(path_Data=config.data_path, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)
    test_dataset = isic_loader(path_Data=config.data_path, train=False, Test=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=config.num_workers,
                             drop_last=True)
    print("train_len:%d".format(config.batch_size * len(train_loader)))
    print(len(test_loader))
    print(len(val_loader))

    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config
    model = seg_net(in_ch=3, start_ch=48, num_classes=model_cfg['num_classes'], img_size=256,
            granular_sizes=[256,128,64], up_factor=[1, 2, 4],
            loops=[1, 2, 3, 4]).cuda()


    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1
    max_iterations = config.epochs * len(train_loader)

    from tqdm import tqdm

    print('#----------Training----------#')
    iterator = tqdm(range(0, config.epochs), ncols=70)
    for epoch in iterator:
        torch.cuda.empty_cache()
        time1 = datetime.now()
        train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            config,
            max_iterations,
            scaler=scaler
        )
        time2 = datetime.now()
        print(f"times:{time2 - time1}=============================")

        if epoch % 400 == 0:
            loss = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
            )
            if loss < min_loss:
                torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                min_loss = loss
                min_epoch = epoch

    torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'latest.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/latest.pth', map_location=torch.device('cpu'))
        model.module.load_state_dict(best_weight)
        loss = test_one_epoch(
            test_loader,
            model,
            criterion,
            logger,
            config,
        )


if __name__ == '__main__':
    config = setting_config
    seeds = [41,42,43]
    for seed in seeds:
        main(config, seed)
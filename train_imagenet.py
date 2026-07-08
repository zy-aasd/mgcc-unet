from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
import logging
from model.launch_seg_imagenet import launch_seg as seg_net
from datasets.dataset_imagenet import *
from sklearn.metrics import accuracy_score
from torch.cuda.amp import autocast, GradScaler


img_size=224
batch_size=64
num_workers=40
learning_rate=0.00035
min_lr = 1e-5

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
# 加载预定义模型
model = seg_net(in_ch=3, start_ch=48, num_classes=1001, img_size=224,
              granular_sizes=[224, 224 // 2, 224 // 4], up_factor=[1, 2, 4],
              loops=[1, 2, 3, 4]).cuda()
# 设置设备为多 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# 如果有多张 GPU, 使用 DataParallel
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)  # 自动在多个 GPU 上分发数据

# 定义损失函数和优化器dataset_imagenet.py
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# device = torch.device('cpu')
# # 训练模型
# logging.info("init model config")
# if not torch.cuda.is_available():
#     device = torch.device("cuda")
#     model = model.to(device)
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# 加载训练集数据集
data_dir = "../../train/"  # 替换为你的训练数据路径
txt_path = './ILSVRC2012_mapping.txt'


train_dataset = CustomImagenetDataset(data_dir, txt_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

# 配置验证集路径
val_dir = "../../val/"  # 替换为你的验证集图像路径
label_file = "../../ILSVRC2012_validation_ground_truth.txt"  # 标签文件路径
# 创建验证数据集和 DataLoader
val_dataset = ImagenetValDataset(val_dir, label_file, transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
scaler = GradScaler()



max_epoch = 100
stp = False
def validate(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    # iteration = 1
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            _, preds = torch.topk(val_outputs, k=5, dim=1)

            #
            # print(f"pred={torch.argmax(torch.softmax(val_outputs[0], dim=0), dim=0)}" +
            #     f", label={val_labels[0]}, iteration={iteration}")
            # iteration += 1
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(val_labels.cpu().numpy())
    top1_acc = accuracy_score(all_labels, [pred[0] for pred in all_preds])
    top5_acc = np.mean([label in pred for label, pred in zip(all_labels, all_preds)])
    print(f"Top-1 Accuracy: {top1_acc:.4f}, Top-5 Accuracy: {top5_acc:.4f}")
    del all_preds, all_labels
    return top1_acc

best_acc = 0.65
item = 0

warmup_epochs = 0
total_steps = max_epoch * len(train_loader)
warmup_steps = warmup_epochs * len(train_loader)


for epoch in range(max_epoch):
    max_iterations = max_epoch * len(train_loader) + 1 # max_epoch = max_iterations // len(trainloader) + 1

    model.train()
    time1 = datetime.now()
    print(f"iteration:{len(train_loader)}")
    avg_loss = []
    for i, (images, labels) in enumerate(train_loader):
        item = item + 1
        images, labels = images.to(device), labels.to(device)


        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()


        if epoch < warmup_steps:
            # warmup
            warmup_factor = item / warmup_steps
            lr_ = learning_rate * warmup_factor
        else:
            lr_ = learning_rate * (1.0 - item / max_iterations) ** 0.9

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        if i % 1000 == 0:
            print(f"Epoch {epoch + 1}, iteration:{i}, Loss: {loss.item():.4f}, pred={torch.argmax(torch.softmax(outputs[0], dim=0), dim=0)}"
                f", label={labels[0]}, lr={lr_}")
        avg_loss.append(loss.item())

        del images, labels, outputs, preds
        torch.cuda.empty_cache()

    time2 = datetime.now()
    avg_lo = np.sum(avg_loss) / len(train_loader)
    print(f"times:{time2-time1}, avg_loss:{avg_lo}=============================")

    #
    if epoch >= 60:
        print("validating.....")
        acc = validate(model, val_loader)
        torch.save(model.state_dict(), f'./pretrained/unet_best_epoch_{epoch}_{acc}.pth')
        print("best model saved.....")

    if epoch == max_epoch - 1:
        # acc = validate(model, val_loader)
        torch.save(model.state_dict(), f'./pretrained/unet_last.pth')
        print("last model saved.....")

    del avg_loss, avg_lo, time1, time2
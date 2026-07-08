import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomImagenetDataset(Dataset):
    def __init__(self, data_dir, label_map_path, transform=None, max_labels=1000):
        self.data_dir = data_dir
        self.transform = transform
        self.max_labels = max_labels

        self.wnid_to_class = self._load_label_map(label_map_path, max_labels)
        self.valid_wnids = set(self.wnid_to_class.keys())

        n = 1.0

        self.image_files = [
            f for f in os.listdir(data_dir)
            if f.endswith(".JPEG") or f.endswith(".jpeg") or f.endswith(".jpg")
        ]

        # sample_size = int(len(self.image_files) * n)
        # self.image_files = random.sample(self.image_files, sample_size)

        self.labels = [f.split('_')[0] for f in self.image_files]
        self.label_indices = [self.wnid_to_class[label] for label in self.labels if label in self.valid_wnids]
    def _load_label_map(self, label_map_path, max_labels):
        wnid_to_class = {}
        with open(label_map_path, 'r', encoding='utf-8-sig') as file:
            for i, line in enumerate(file):
                if i >= max_labels:  # 仅读取前 max_labels 行
                    break
                cls_id, wnid = line.strip().split()
                wnid_to_class[wnid] = int(cls_id)  # 确保标签是从 1 开始的
        return wnid_to_class

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        label = self.label_indices[idx]

        # 加载图像
        img_path = os.path.join(self.data_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {img_name} -> {e}")
            return None, None

        if self.transform:
            image = self.transform(image)

        return image, label




from torchvision.datasets.folder import default_loader  # 用于加载图像

# 定义自定义的验证数据集
class ImagenetValDataset(Dataset):
    def __init__(self, val_dir, label_file, transform=None):
        """
        :param val_dir: 验证集图像文件夹路径
        :param label_file: 验证集标签文件路径
        :param transform: 数据增强/预处理
        """
        self.val_dir = val_dir
        self.transform = transform
        # 加载标签
        with open(label_file, 'r') as f:
            self.labels = [int(line.strip()) for line in f]
        # 加载图像文件名并排序
        self.image_files = sorted(os.listdir(val_dir))  # 假设文件已按自然顺序命名
        assert len(self.image_files) == len(self.labels), "标签数量与图像文件数量不匹配"
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        # 获取图像路径和对应标签
        img_path = os.path.join(self.val_dir, self.image_files[idx])
        label = self.labels[idx] # 标签从 1-based
        # 加载图像
        image = default_loader(img_path)
        # 应用数据增强
        if self.transform:
            image = self.transform(image)
        return image, label



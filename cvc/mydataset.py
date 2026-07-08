import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import os
from PIL import Image, ImageOps
import imageio.v2 as imageio
import tifffile

import os
from PIL import Image


def check_dataset(root_path, dataset_type='train'):
    """检查数据集完整性"""
    img_path = os.path.join(root_path, f"datamodel/{dataset_type}/images/")
    lb_path = os.path.join(root_path, f"datamodel/{dataset_type}/labels/")

    images = os.listdir(img_path)
    labels = os.listdir(lb_path)

    print(f"Images: {len(images)}, Labels: {len(labels)}")

    # 检查每个文件
    for i, (img_file, lbl_file) in enumerate(zip(images, labels)):
        img_full = os.path.join(img_path, img_file)
        lbl_full = os.path.join(lb_path, lbl_file)

        try:
            # 尝试打开图像
            img = Image.open(img_full)
            lbl = Image.open(lbl_full)
            print(f"✓ {i}: {img_file} ({img.size}) - {lbl_file} ({lbl.size})")
        except Exception as e:
            print(f"✗ {i}: Error - {img_file}: {e}")

# 运行检查
check_dataset("../../../data/cvc_clinicdb/")


class CVCClinicDB_Dataset(data.Dataset):
    def __init__(self, root=None, dataset_type='train', transform=None):
        self.dataset_type = dataset_type
        self.transform = transform

        self.img_path = root + "datamodel/{}/images/".format(dataset_type)
        self.lb_path = root + "datamodel/{}/labels/".format(dataset_type)
        self.item_image = os.listdir(root + "datamodel/{}/images/".format(dataset_type))
        self.item_gt = os.listdir(root + "datamodel/{}/labels/".format(self.dataset_type))

    def __getitem__(self, index):
        img_name = tifffile.imread(self.img_path + self.item_image[index])
        label_name = tifffile.imread(self.lb_path + self.item_gt[index])

        img_name = np.array(img_name)
        label_name = np.array(label_name)
        label_name = np.where(label_name>200, 1, 0)

        image = Image.fromarray(np.uint8(img_name))
        label = Image.fromarray(np.uint8(label_name), 'L')


        if self.transform:
            image,label = self.transform(image, label)
        return image,label

    def __len__(self):
        return len(self.item_image)
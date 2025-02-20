import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import pandas as pd

from utils import random_click  # 假设你已经有 random_click 等工具函数


class RSNA(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='click', plane=False):
        """
        Args:
            args: 参数字典，包含如 image_size 等信息
            data_path: 数据路径，包含图片和mask的文件夹
            transform: 图像数据的预处理（可选）
            transform_msk: mask数据的预处理（可选）
            mode: 'Training' 或 'Testing'，决定是否进行数据增强等处理
            prompt: 'click' 或其他，表示用户输入的提示方式
            plane: 是否使用平面数据（可选）
        """
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size  # 目标图像大小
        self.transform = transform
        self.transform_msk = transform_msk

        # 获取图片和mask文件名列表
        df = pd.read_csv(os.path.join(data_path, 'annotations'+  '_' + mode +'.csv'), encoding='gbk')
        self.name_list = df.iloc[:, 0].tolist()  # 假设文件名在第一列
        self.label_list = df.iloc[:, 1].tolist()  # 假设标签在第二列

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """获取单个样本，包括图像和mask"""
        point_label = 1  # 假设每个样本的point_label初始为1

        # 获取图片和mask路径
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, 'image', name)  # 假设图片存储在 'image' 文件夹
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, 'mask', mask_name)  # 假设mask存储在 'mask' 文件夹

        # 读取图像和mask
        img = Image.open(img_path).convert('RGB')  # 转换为RGB格式
        mask = Image.open(msk_path).convert('L')  # 转换为单通道灰度图

        # 图像和mask的预处理
        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)  # 重设mask的大小以匹配图像

        # 如果需要点击提示
        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask) / 255, point_label)

        # 应用图像和mask的变换
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)  # 应用图像的预处理
            torch.set_rng_state(state)

        if self.transform_msk:
            mask = self.transform_msk(mask).int()  # 应用mask的预处理

        # 获取文件名并保存到meta字典中
        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj': name}

        return {
            'image': img,  # [C, H, W] 格式的图像
            'label': mask,  # [1, H, W] 格式的mask
            'p_label': point_label,  # 提示标签
            'pt': pt,  # 提示位置，可能是点击坐标
            'image_meta_dict': image_meta_dict  # 包含文件名的meta信息
        }

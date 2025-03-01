import torch
from torchvision import models

# 定义模型
num_classes = 2  # 根据您的任务设置类别数（包含背景）
model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载训练好的模型权重
model.load_state_dict(torch.load("fcn_resnet50_segmentation.pth", map_location=device))
model.eval()

from torchvision import transforms
from PIL import Image
import numpy as np
import torch

# 图像预处理：调整尺寸为 256x256，并归一化
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = test_transform(image)
    return image.unsqueeze(0)  # 增加批次维度


import os
import cv2

import os
import cv2
import numpy as np

def infer_and_save_cropped(model, image_path, output_dir, device, expand_ratio=0.03):
    # 预处理图像
    input_tensor = preprocess_image(image_path).to(device)

    # 推理
    with torch.no_grad():
        output = model(input_tensor)['out']
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # 将预测的掩码调整回原始图像尺寸
    original_image = Image.open(image_path).convert("RGB")
    orig_width, orig_height = original_image.size
    mask_resized = cv2.resize(pred_mask, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)

    # 查找掩码中的前景区域
    foreground_indices = np.argwhere(mask_resized == 1)
    if foreground_indices.size == 0:
        print(f"No foreground detected in {image_path}. Skipping.")
        return

    # 计算边界框
    y_min, x_min = foreground_indices.min(axis=0)
    y_max, x_max = foreground_indices.max(axis=0)

    # 计算扩展距离
    expand_height = int(expand_ratio * orig_height)
    expand_width = int(expand_ratio * orig_width)

    # 扩展边界框，并确保不超出图像边界
    x_min = max(x_min - expand_width, 0)
    x_max = min(x_max + expand_width, orig_width)
    y_min = max(y_min - expand_height, 0)
    y_max = min(y_max + expand_height, orig_height)

    # 裁剪并保存扩展后的图像
    cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cropped_image.save(output_path)
    print(f"Saved cropped image to {output_path}")

    
    
import glob

test_image_dir = "/private/workspace/cyt/bone_age_assessment/data/data_yuwei/val"  # 替换为您的测试集图像文件夹路径
output_dir = "/private/workspace/cyt/bone_age_assessment/data/data_yuwei/val_clean"     # 替换为您希望保存结果的文件夹路径

test_image_paths = glob.glob(os.path.join(test_image_dir, "*.*"))


for image_path in test_image_paths:
    infer_and_save_cropped(model, image_path, output_dir, device)




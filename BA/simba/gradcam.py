# -*- coding: utf-8 -*-
"""
Bone Age Assessment SIMBA test routine with Grad-CAM visualization.
"""

import os
import cv2
import numpy as np
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable

# Local imports
from models.simba import SIMBA
from data.data_loader import BoneageDataset as Dataset
from utils import AverageMeter

# Argument parser
parser = argparse.ArgumentParser()

# Model settings
parser.add_argument('--relative-age', default=False, action='store_true')
parser.add_argument('--chronological-age', default=False, action='store_true')
parser.add_argument('--gender-multiplier', default=False, action='store_true')
parser.add_argument('--use-gut-microbiome', default=False, action='store_true')
parser.add_argument('--use-pe-performance', default=False, action='store_true')
parser.add_argument('--use-correlation', default=False, action='store_true')
parser.add_argument('--use-image', default=False, action='store_true',
                help='Train model with image')


parser.add_argument('--feature-extractor', default='resnet', type=str,
                help='imaage feature extraction')

# Dataloading settings
parser.add_argument('--dataset', default='RSNA', type=str, choices=['RSNA', 'RHPE', 'KG'])
parser.add_argument('--data-test', default='data/test/images', type=str)
parser.add_argument('--heatmaps-test', default='data/test/heatmaps', type=str)
parser.add_argument('--ann-path-test', default='test.csv', type=str)
parser.add_argument('--rois-path-test', default='test.json', type=str)

# Output settings
parser.add_argument('--save-folder', default='output/', type=str)
parser.add_argument('--snapshot', default='boneage_bonet_weights.pth', type=str)
parser.add_argument('--save-file', default='test_results.csv', type=str)

# System settings
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--batch-size', default=1, type=int)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--gpu', type=str, default='0')

args = parser.parse_args()
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Create output folder
gradcam_folder = os.path.join(args.save_folder, 'gradcam')
os.makedirs(gradcam_folder, exist_ok=True)

# Create the network architecture and load the best model
net = SIMBA(
    chronological_age=args.chronological_age,
    gender_multiplier=args.gender_multiplier,
    use_gut_microbiome=args.use_gut_microbiome,
    use_pe_performance=args.use_pe_performance,
    use_correlation=args.use_correlation,
    use_image=args.use_image,
    feature_extractor=args.feature_extractor
).cuda()

# Load model weights
if os.path.exists(args.snapshot):
    print(f'Loading model from: {args.snapshot}')
    net.load_state_dict(torch.load(args.snapshot, map_location=device))

# Loss function
criterion = nn.L1Loss()

# Data transformations
test_transform = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.ToTensor()
])

# DataLoader
test_dataset = Dataset([args.data_test], [args.heatmaps_test],
                        [args.ann_path_test], [args.rois_path_test],
                        img_transform=test_transform, dataset=args.dataset)

test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册前向和反向 hook
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, gender, chronological_age, gut, pe, cor):
        self.model.zero_grad()
        output = self.model(input_tensor, gender, chronological_age, gut, pe, cor)
        
        if isinstance(output, tuple):  # 可能有辅助分类器
            output = output[0]

        # class_score = output[:, target_class]  # 获取目标类别的预测值
        # class_score = output.mean()
        class_score = output[:, 0]  # 取第一个通道的输出作为目标
        class_score.backward()  # 计算梯度

        # 计算 Grad-CAM 权重
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])  # GAP
        activations = self.activations[0]  # 取 batch 的第一张图

        # 计算加权特征图
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU 激活
        heatmap /= np.max(heatmap)  # 归一化到 0-1
        return heatmap



import os
import cv2
import numpy as np

def apply_heatmap(img, heatmap, save_dir, p_id, blur_ksize=5):
    """
    叠加 Grad-CAM 热力图并保存，同时保存原始归一化图像
    
    参数:
        img: 原始灰度图像 (numpy array)
        heatmap: 热力图 (numpy array)
        save_dir: 保存目录
        p_id: 图片 ID
        blur_ksize: 高斯模糊核大小，默认 5，可调整

    返回:
        superimposed_img: 叠加后的 Grad-CAM 结果
    """

    # 1. **归一化 img 并转换为 uint8**
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # 避免除零
    img_uint8 = np.uint8(255 * img)  # 转换到 0-255

    # 2. **转换为 BGR 3 通道**
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

    # 3. **调整 heatmap 大小**
    heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))  # 调整 heatmap 大小

    # 4. **最大最小归一化 heatmap**
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)  # 归一化到 0-1
    heatmap = np.uint8(255 * heatmap)  # 转换到 0-255

    # 5. **应用高斯模糊平滑 heatmap**
    if blur_ksize > 1:
        heatmap = cv2.GaussianBlur(heatmap, (blur_ksize, blur_ksize), 0)

    # 6. **伪彩色映射**
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 7. **确保 img 和 heatmap 形状匹配**
    assert img_bgr.shape == heatmap_color.shape, f"Shape mismatch: img {img_bgr.shape}, heatmap {heatmap_color.shape}"

    # 8. **叠加 Grad-CAM 热力图**
    superimposed_img = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)

    # 9. **创建保存目录**
    save_path = os.path.join(save_dir, 'gradcam')
    os.makedirs(save_path, exist_ok=True)

    # 10. **保存原图和叠加热力图**
    img_save_path = os.path.join(save_path, f"{p_id}_original.png")
    gradcam_save_path = os.path.join(save_path, f"{p_id}.png")

    cv2.imwrite(img_save_path, img_uint8)  # 保存原始归一化图像
    cv2.imwrite(gradcam_save_path, superimposed_img)  # 保存 Grad-CAM 叠加图

    print(f"Original image saved at: {img_save_path}")
    print(f"Grad-CAM image saved at: {gradcam_save_path}")

    return superimposed_img







# ---------------- Main Inference Loop ----------------
def test(model, dataloader, criterion):
    model.eval()
    epoch_loss = AverageMeter()
    results = {}
    save_path = args.save_folder
    with torch.enable_grad():
        for i, batch in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
            inputs, bone_ages, gender, chronological_age, p_id, gut, pe, cor = batch
            inputs, gender, chronological_age, gut, pe, cor= Variable(inputs).cuda(), Variable(gender).cuda(), Variable(chronological_age).cuda(), Variable(gut).cuda(), Variable(pe).cuda(), Variable(cor).cuda()
            # 选择目标层：Mixed_7c 是 SIMBA 的最后一个 Inception 层
            target_layer = model.feature_extractor.layer2[0] # layer1[-1]， layer2[-1]， layer3[-1], layer2[0]
            grad_cam = GradCAM(model, target_layer) # Mixed_7c, Mixed_7a, Mixed_6c, Mixed_6a, Mixed_5b, Mixed_5c, Conv2d_5a_1x1, Conv2d_4a_3x3[0], Conv2d_1a_3x3[0]
            # Conv2d_5a_1x1比较合适多模态的热力图，Mixed_5b以后就没有热力区域了
            

            # 生成 Grad-CAM 热力图
            heatmap = grad_cam.generate_cam(inputs, gender, chronological_age, gut, pe, cor)
            
            # 1. 将 PyTorch Tensor 转换为 NumPy
            img = inputs.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (C, H, W) → (H, W, C)
            img = img[..., 0]  # **只取第一个通道，变成 (H, W)**

            # 叠加到原图
            result = apply_heatmap(img, heatmap, save_path, str(int(p_id.item())))


    # print(f"Test Loss: {epoch_loss.avg}")
    return results

def main():
    print("Starting Inference...")
    test_results = test(net, test_loader, criterion)
    df = pd.DataFrame.from_dict(test_results, orient="index", columns=["Predicted Bone Age"])
    df.to_csv(os.path.join(args.save_folder, args.save_file))
    print(f"Results saved to {args.save_folder}/{args.save_file}")

if __name__ == '__main__':
    main()

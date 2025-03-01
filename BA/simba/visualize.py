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

# Load the model
net = SIMBA(
    chronological_age=args.chronological_age,
    gender_multiplier=args.gender_multiplier,
    use_gut_microbiome=args.use_gut_microbiome,
    use_pe_performance=args.use_pe_performance,
    use_correlation=args.use_correlation
).to(device)

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

# ---------------- Grad-CAM Implementation ----------------

def gradcam_visualization(model, inputs, gender, chronological_age, gut, pe, cor, save_path):
    """
    计算 Grad-CAM 热力图并保存
    """
    model.eval()
    
    # 选取模型中的最后一层 Inception 模块
    target_layer = model.Mixed_5c

    # 钩子函数：用于获取前向传播的特征图
    feature_maps = []
    def forward_hook(module, input, output):
        feature_maps.append(output)
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    
    # 绑定 backward hook，获取 `gradients`
    gradients = []
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)
    
    # 前向传播计算预测结果
    inputs.requires_grad_(True)
    outputs = model(inputs, gender, chronological_age, gut, pe, cor)  # 添加额外参数
    outputs.requires_grad_(True)
    outputs = outputs.squeeze()
    
    # 根据输出元素数量判断是否为标量（回归任务通常输出标量）
    if outputs.numel() == 1:
        outputs.backward(retain_graph=True)
    else:
        print("警告：输出不是标量，使用输出均值进行反向传播")
        outputs.mean().backward(retain_graph=True)
    
    # 检查钩子是否捕获数据
    if len(gradients) == 0 or len(feature_maps) == 0:
        handle_fwd.remove()
        handle_bwd.remove()
        raise RuntimeError("钩子没有捕获到梯度或特征图，请检查目标层的选择。")
    
    # 获取第一个捕获的特征图和梯度
    grads_val = gradients[0]
    fmap = feature_maps[0]
    
    # 可选：打印调试信息
    # print("Feature maps shape:", fmap.shape)
    # print("Gradients shape:", grads_val.shape)
    
    # 计算权重并生成 Grad-CAM
    weights = grads_val.mean(dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * fmap, dim=1).squeeze().cpu().detach().numpy()
    cam = np.maximum(cam, 0)  # ReLU激活
    if cam.max() != 0:
        cam = cam / (cam.max() + 1e-8)  # 归一化

    # 还原热力图大小
    cam = cv2.resize(cam, (500, 500), interpolation=cv2.INTER_LINEAR)
    cam = np.uint8(255 * cam)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    # 读取原始 X 光影像
    img = inputs[0][0].cpu().detach().numpy().squeeze()
    img = np.uint8(255 * (img - img.min()) / (img.max() - img.min() + 1e-8))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 叠加热力图
    overlay = cv2.addWeighted(img, 0.5, cam, 0.5, 0)
    
    # 保存 Grad-CAM 结果
    cv2.imwrite(save_path, overlay)

    # 移除钩子
    handle_fwd.remove()
    handle_bwd.remove()



# ---------------- Main Inference Loop ----------------
def test(model, dataloader, criterion):
    model.eval()
    epoch_loss = AverageMeter()
    results = {}
    with torch.enable_grad():
        for i, batch in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
            inputs, bone_ages, gender, chronological_age, p_id, gut, pe, cor = batch
            inputs, gender, chronological_age, gut, pe, cor= Variable(inputs).cuda(), Variable(gender).cuda(), Variable(chronological_age).cuda(), Variable(gut).cuda(), Variable(pe).cuda(), Variable(cor).cuda()
            bone_ages = Variable(bone_ages).cuda()

            # 清除梯度
            model.zero_grad()
            
            # with torch.set_grad_enabled(True):
            #     outputs = model(inputs, gender, chronological_age, gut, pe, cor)
            
            # # 计算损失（禁用梯度计算）
            # with torch.no_grad():
            #     # 计算损失
            #     relative_ages = chronological_age.squeeze(1) - bone_ages
            #     loss = criterion(outputs.squeeze(), relative_ages)
            #     epoch_loss.update(loss)

            # Grad-CAM 处理（确保传递所有参数）
            gradcam_path = f"{args.save_folder}/gradcam/gradcam_{p_id.item()}.png"
            gradcam_visualization(model, inputs, gender, chronological_age, gut, pe, cor, gradcam_path)

        # results[p_id.item()] = outputs.item()

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

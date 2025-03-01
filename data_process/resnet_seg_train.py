import os
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import cv2

# ----------------- 自定义数据集 -----------------
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        """
        :param image_dir: 图像文件夹路径（RGB 图片）
        :param mask_dir: 分割 mask 文件夹路径（单通道，每个像素的值代表类别）
        :param transform: 针对图像的预处理
        :param mask_transform: 针对 mask 的预处理（注意 mask 一般需要用最近邻插值）
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        # 按文件名排序，确保图像与 mask 一一对应
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.*")))
        self.mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.*")))
        assert len(self.image_files) == len(self.mask_files), "图像与 mask 数量不一致"
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像与 mask
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert('L')
        
        if self.transform is not None:
            image = self.transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask).int().long()
        else:
            # 默认转换：转为 numpy 数组，再转为 LongTensor（像素值作为类别标签）
            mask = np.array(mask)
            mask = torch.from_numpy(mask).int().long()
        mask = torch.from_numpy(np.array(mask)).long() // 255
        
        # 检查 mask 的值范围
        min_val = mask.min().item()
        max_val = mask.max().item()
        assert min_val >= 0 and max_val < num_classes, f"Mask values should be in [0, {num_classes-1}], but got min={min_val}, max={max_val}"
        
        return image, mask

# ----------------- 数据预处理 -----------------
# 图像预处理：统一调整尺寸为 256x256，并归一化
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# mask预处理，使用最近邻插值保持类别信息
mask_transform = lambda x: torch.from_numpy(np.array(x.resize((256,256), resample=Image.NEAREST))).long()

# ----------------- 数据集与 DataLoader -----------------
# 修改下面路径为你的实际文件夹路径
train_image_dir = "/private/workspace/cyt/bone_age_assessment/data/RSNA/segmentation/image"
train_mask_dir = "/private/workspace/cyt/bone_age_assessment/data/RSNA/segmentation/mask"

dataset = SegmentationDataset(train_image_dir, train_mask_dir,
                              transform=train_transform,
                              mask_transform=mask_transform)

# 划分训练集和验证集：验证集占 20%
total_size = len(dataset)
val_size = int(0.2 * total_size)
train_size = total_size - val_size
train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, num_workers=4)

# ----------------- 模型构建 -----------------
# 使用 torchvision 提供的 FCN 模型，backbone 为 ResNet50
num_classes = 2  # 根据任务设置类别数（包含背景）
model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ----------------- 损失函数与优化器 -----------------
criterion = nn.CrossEntropyLoss()  # 目标 mask 为 LongTensor，每个像素为类别标签
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ----------------- 辅助函数：反归一化 -----------------
def unnormalize(tensor, mean, std):
    """
    将归一化的 tensor 反归一化，tensor shape: (C, H, W)
    """
    for t, m, s in zip(tensor, mean, std):
        tensor[tensor==t]  # 仅作占位（无实际作用）
    # 直接利用广播进行反归一化
    return tensor * std.view(-1,1,1) + mean.view(-1,1,1)

# ----------------- 辅助函数：可视化并保存 -----------------
def visualize_predictions(model, loader, device, epoch, vis_dir="visualization", max_samples=4):
    os.makedirs(vis_dir, exist_ok=True)
    model.eval()
    # 取验证集中的一个 batch
    images, masks = next(iter(loader))
    images = images.to(device)
    masks = masks.to(device)
    with torch.no_grad():
        outputs = model(images)['out']
        preds = torch.argmax(outputs, dim=1)  # shape: [B, H, W]
    
    # 反归一化原图：还原到 [0,1]
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(3,1,1)
    images_unnorm = images * std + mean

    images_np = images_unnorm.cpu().numpy()
    masks_np = masks.cpu().numpy()
    preds_np = preds.cpu().numpy()

    # 对于每个样本，将原图、GT mask、预测 mask 拼接为一张图（横向拼接）
    for i in range(min(max_samples, images_np.shape[0])):
        # 原图转换为 H x W x C，并转为 0~255 uint8
        orig = np.transpose(images_np[i], (1,2,0))  # (H, W, C)
        orig = (orig * 255).clip(0,255).astype(np.uint8)
        # 将 GT mask 和预测 mask 转为彩色图（这里简单将 0 映射为黑，1 映射为白）
        gt = (masks_np[i] * 255).astype(np.uint8)
        pred = (preds_np[i] * 255).astype(np.uint8)
        gt_color = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
        pred_color = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
        # 拼接：原图 | GT | 预测
        composite = np.concatenate([orig, gt_color, pred_color], axis=1)
        save_path = os.path.join(vis_dir, f"epoch_{epoch+1}_sample_{i}.jpg")
        cv2.imwrite(save_path, composite)
    model.train()

# ----------------- 训练与验证循环 -----------------
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)['out']  # 输出 shape: [B, num_classes, H, W]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss /= train_size

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
    val_loss /= val_size

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    # 每隔 5 个 epoch 可视化一次验证集样本
    if (epoch + 1) % 5 == 0:
        visualize_predictions(model, val_loader, device, epoch, vis_dir="visualization", max_samples=4)

# ----------------- 保存模型 -----------------
torch.save(model.state_dict(), "fcn_resnet50_segmentation.pth")
print("模型保存完毕！")

import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50
from segmentation_models_pytorch import Unet, DeepLabV3, DeepLabV3Plus
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import cv2
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF



# ----------------- 早停策略 -----------------
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  
            return False  
        else:
            self.counter += 1
            print(f"Early Stopping Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print("Early stopping triggered.")
                return True  
            return False

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
        self.image_files = self.image_files[:100]
        self.mask_files = self.mask_files[:100]
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


import cv2
import numpy as np
import torch

def compute_edge_accuracy(preds, masks):
    """
    计算 Edge Accuracy
    :param preds: 预测的 mask (B, H, W) -> 0/1
    :param masks: 真实的 mask (B, H, W) -> 0/1
    :return: Edge Accuracy
    """
    kernel = np.ones((3, 3), np.uint8)  # 3x3 结构元素
    preds_np = preds.cpu().numpy().astype(np.uint8)
    masks_np = masks.cpu().numpy().astype(np.uint8)

    edge_accs = []
    
    for i in range(preds.shape[0]):  # 遍历 batch 中的每张图片
        mask_edge = cv2.morphologyEx(masks_np[i], cv2.MORPH_GRADIENT, kernel)  # 真实边缘
        pred_edge = cv2.morphologyEx(preds_np[i], cv2.MORPH_GRADIENT, kernel)  # 预测边缘

        # 计算边缘区域的匹配度
        edge_intersection = np.logical_and(mask_edge, pred_edge).sum()
        edge_union = np.logical_or(mask_edge, pred_edge).sum()

        if edge_union == 0:
            edge_acc = 1.0  # 如果没有边缘，视为完全正确
        else:
            edge_acc = edge_intersection / edge_union  # 计算 IoU 形式的匹配度

        edge_accs.append(edge_acc)

    return np.mean(edge_accs)  # 返回 batch 的平均值


def compute_metrics(preds, masks):
    """
    计算 IoU, Dice, Edge Accuracy
    """
    smooth = 1e-6
    preds_bin = preds > 0.5  

    # IoU 计算
    intersection = torch.logical_and(preds_bin, masks).sum(dim=(1,2))  
    union = torch.logical_or(preds_bin, masks).sum(dim=(1,2))  
    iou = (intersection + smooth) / (union + smooth)  

    # Dice 计算
    dice = (2 * intersection + smooth) / (preds_bin.sum(dim=(1,2)) + masks.sum(dim=(1,2)) + smooth)  

    # Edge Accuracy 计算
    edge_acc = compute_edge_accuracy(preds_bin, masks)

    return iou.mean().item(), dice.mean().item(), edge_acc


# ----------------- 训练过程可视化 -----------------
def visualize_predictions(model, loader, device, epoch, vis_dir="visualization", max_samples=5):
    os.makedirs(vis_dir, exist_ok=True)
    model.eval()
    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        # FCN / DeepLabV3 返回字典，需要 `["out"]`
        if isinstance(outputs, dict):  
            outputs = outputs["out"]
        # SegFormer / SETR 可能有 `logits`
        elif hasattr(outputs, "logits"):  
            outputs = outputs.logits  # 适用于 SegFormer / SETR
        preds = torch.argmax(outputs, dim=1)

    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(3,1,1)
    images_unnorm = images * std + mean

    images_np = images_unnorm.cpu().numpy()
    masks_np = masks.cpu().numpy()
    preds_np = preds.cpu().numpy()

    for i in range(min(max_samples, images_np.shape[0])):
        orig = np.transpose(images_np[i], (1,2,0)) * 255
        orig = orig.clip(0,255).astype(np.uint8)
        gt = (masks_np[i] * 255).astype(np.uint8)
        pred = (preds_np[i] * 255).astype(np.uint8)
        gt_color = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
        pred_color = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
        composite = np.concatenate([orig, gt_color, pred_color], axis=1)
        save_path = os.path.join(vis_dir, f"epoch_{epoch+1}_sample_{i}.jpg")
        cv2.imwrite(save_path, composite)
        
# ----------------- 选择分割模型 -----------------
def get_segmentation_model(model_name, num_classes):
    """
    获取分割模型
    :param model_name: 模型名称 ["fcn", "deeplabv3", "unet", "setr", "segformer"]
    :param num_classes: 分割类别数
    :return: 选择的模型
    """
    if model_name == "fcn":
        model = fcn_resnet50(pretrained=False, num_classes=num_classes)
    elif model_name == "deeplabv3":
        model = deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
    elif model_name == "unet":
        model = Unet(encoder_name="resnet34", classes=num_classes, encoder_weights=None)
    # elif model_name == "setr":
    #     model = SETRModel.from_pretrained("nvidia/setr-pup", num_labels=num_classes)
    elif model_name == "segformer":
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512", num_labels=num_classes)
    else:
        raise ValueError(f"❌ 不支持的模型名称: {model_name}")
    
    return model



# ----------------- 训练 -----------------
def train_segmentation(model_name, train_loader, val_loader, num_classes, device, num_epochs=100):
    model = get_segmentation_model(model_name, num_classes).to(device)
    # ----------------- 损失函数与优化器 -----------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # **余弦退火学习率调度器**
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    # ----------------- 训练 -----------------
    num_epochs = 100
    early_stopping = EarlyStopping(patience=10)
    best_val_loss = float('inf')

    # ----------------- TensorBoard -----------------
    writer = SummaryWriter(log_dir="runs/segmentation_experiment")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # FCN / DeepLabV3 返回字典，需要 `["out"]`
            if isinstance(outputs, dict):  
                outputs = outputs["out"]
            # SegFormer / SETR 可能有 `logits`
            elif hasattr(outputs, "logits"):  
                outputs = outputs.logits  # 适用于 SegFormer / SETR
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)
        scheduler.step()

        model.eval()
        val_loss, iou_total, dice_total, edge_acc_total = 0, 0, 0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                # FCN / DeepLabV3 返回字典，需要 `["out"]`
                if isinstance(outputs, dict):  
                    outputs = outputs["out"]
                # SegFormer / SETR 可能有 `logits`
                elif hasattr(outputs, "logits"):  
                    outputs = outputs.logits  # 适用于 SegFormer / SETR
                preds = torch.argmax(outputs, dim=1)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                iou, dice, edge_acc = compute_metrics(preds, masks)
                iou_total += iou
                dice_total += dice
                edge_acc_total += edge_acc

        mean_iou = iou_total / len(val_loader)
        mean_dice = dice_total / len(val_loader)
        mean_edge_acc = edge_acc_total / len(val_loader)
        val_loss /= val_size

        writer.add_scalar("IoU/val", mean_iou, epoch)
        writer.add_scalar("Dice/val", mean_dice, epoch)
        writer.add_scalar("EdgeAcc/val", mean_edge_acc, epoch)

        print(f"Epoch {epoch+1}: IoU={mean_iou:.4f}, Dice={mean_dice:.4f}, EdgeAcc={mean_edge_acc:.4f}")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"checkpoint/best_{model_name}.pth")
            print(f"✔ Model saved at epoch {epoch+1} with val_loss: {val_loss:.4f}")

        # 每 3 个 epoch 可视化一次
        if (epoch + 1) % 3 == 0:
            visualize_predictions(model, val_loader, device, epoch, vis_dir=f"visualization/{model_name}", max_samples=10)

        if early_stopping(val_loss):
            break

    writer.close()

# ----------------- 运行训练 -----------------
if __name__ == "__main__":
    
    # 训练数据
    # ----------------- 数据预处理 -----------------
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    mask_transform = lambda x: torch.from_numpy(np.array(x.resize((256,256), resample=Image.NEAREST))).long()

    # ----------------- 数据加载 -----------------
    train_image_dir = "/private/workspace/cyt/bone_age_assessment/data/RSNA/segmentation/image"
    train_mask_dir = "/private/workspace/cyt/bone_age_assessment/data/RSNA/segmentation/mask"

    dataset = SegmentationDataset(train_image_dir, train_mask_dir, transform=train_transform, mask_transform=mask_transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, num_workers=4)

    # ----------------- 模型构建 -----------------
    num_classes = 2  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 选择模型（可选："fcn", "deeplabv3", "unet", "setr", "segformer"）
    model_name = "segformer"

    train_segmentation(model_name, train_loader, val_loader, num_classes, device)

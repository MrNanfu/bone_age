import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 定义数据集类
class BoneAgeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 定义模型
def get_resnet_model(num_classes=1):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # 添加 Dropout
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model

# 训练函数
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# 验证函数
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            
            running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# 加载数据
train_ann_file = '/private/workspace/cyt/bone_age_assessment/data/data_yuwei/annotations/train_ann.csv'
annotations_filter = pd.read_csv(train_ann_file)

img_dir = '/private/workspace/cyt/bone_age_assessment/data/data_yuwei/train'
img_extensions = ['.png', '.jpg', '.bmp']

image_paths = []
labels = []

for i, row in annotations_filter.iterrows():
    for ext in img_extensions:
        img_path = os.path.join(img_dir, str(int(row['ID'])) + ext)
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(row['Boneage'])
            break

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建数据集
dataset = BoneAgeDataset(image_paths, labels, transform=transform)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建 DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=12, shuffle=False)

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = get_resnet_model().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 添加 L2 正则化
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# 训练循环
num_epochs = 30
best_val_loss = float('inf')
patience = 5
early_stop_counter = 0

for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, criterion, optimizer, device)
    val_loss = validate(model, val_dataloader, criterion, device)
    
    # 更新学习率
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # 早停法逻辑
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)  # 使用 register_full_backward_hook
    
    def save_activations(self, module, input, output):
        self.activations = output
        print("Activations shape:", self.activations.shape)  # 调试信息
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        print("Gradients shape:", self.gradients.shape)  # 调试信息
    
    def forward(self, x):
        return self.model(x)
    
    def backward(self, outputs):
        self.model.zero_grad()
        outputs.backward(torch.ones_like(outputs))
    
    def generate(self, x):
        self.model.eval()
        outputs = self.forward(x)
        self.backward(outputs)
        
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        heatmap = torch.mul(self.activations, pooled_gradients).sum(dim=1, keepdim=True)
        heatmap = nn.functional.relu(heatmap)
        heatmap /= torch.max(heatmap)  # 归一化到 [0, 1]
        
        return heatmap.squeeze().cpu().detach().numpy()

# 使用Grad-CAM生成热力图
target_layer = model.layer4[-1].conv3
grad_cam = GradCAM(model, target_layer)

# 检查输入图像
image = Image.open("/private/workspace/cyt/bone_age_assessment/data/data_yuwei/train/104.png").convert('RGB')
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')
plt.show()

# 检查输入张量
input_tensor = transform(image).unsqueeze(0).to(device)
print("Input tensor shape:", input_tensor.shape)

# 检查模型输出
output = model(input_tensor)
print("Model output:", output)

# 检查 Grad-CAM 的热力图
heatmap = grad_cam.generate(input_tensor)
print("Heatmap shape:", heatmap.shape)
print("Heatmap min:", heatmap.min(), "max:", heatmap.max())

# 检查热力图的值
print("Heatmap min:", heatmap.min(), "max:", heatmap.max())

# 如果热力图的值过小，可以尝试放大
if heatmap.max() < 1e-3:
    print("Heatmap values are too small!")
    heatmap = heatmap * 100  # 放大热力图的值

# 调整热力图大小并应用颜色映射
heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
print("Heatmap after colormap shape:", heatmap.shape)

# 将热力图从 BGR 转换为 RGB
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


# 叠加热力图和原图
superimposed_img = heatmap * 0.4 + np.array(image)
superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)  # 确保值在 [0, 255] 范围内
print("Superimposed image shape:", superimposed_img.shape)
print("Superimposed image min:", superimposed_img.min(), "max:", superimposed_img.max())
print("Superimposed image dtype:", superimposed_img.dtype)

# 显示叠加后的图像
plt.imshow(superimposed_img)
plt.title("Superimposed Image")
plt.axis('off')
plt.show()

# 保存叠加后的图像
output_path = "superimposed_image.jpg"  # 保存路径
cv2.imwrite(output_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
print(f"Image saved to {output_path}")
import torch
import numpy as np
import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import inspect
from models.simba import SIMBA
from data.data_loader import BoneageDataset as Dataset

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='KG', type=str, choices=['RSNA', 'RHPE', 'KG'])
parser.add_argument('--data-test', default='data/test/images', type=str)
parser.add_argument('--heatmaps-test', default='data/test/heatmaps', type=str)
parser.add_argument('--ann-path-test', default='test.csv', type=str)
parser.add_argument('--rois-path-test', default='test.json', type=str)
parser.add_argument('--save-folder', default='output/', type=str)
parser.add_argument('--snapshot', default='boneage_bonet_weights.pth', type=str)
parser.add_argument('--batch-size', default=1, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--gpu', type=str, default='0')

parser.add_argument('--feature-extractor', default='resnet', type=str,
                help='imaage feature extraction')

# **动态多模态选择（用于计算贡献矩阵）**
parser.add_argument('--relative-age', default=False, action='store_true')
parser.add_argument('--chronological-age', default=False, action='store_true')
parser.add_argument('--gender-multiplier', default=False, action='store_true')
parser.add_argument('--use-image', action='store_true')
parser.add_argument('--use-gut-microbiome', action='store_true')
parser.add_argument('--use-pe-performance', action='store_true')
parser.add_argument('--use-correlation', action='store_true')
parser.add_argument('--use-gender', default=False, action='store_true',
                help='Train model with gender')

args = parser.parse_args()
print(args)

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


# 加载模型
model = SIMBA(
    chronological_age=args.chronological_age,
    use_gender=args.use_gender,
    gender_multiplier=args.gender_multiplier,
    use_gut_microbiome=args.use_gut_microbiome,
    use_pe_performance=args.use_pe_performance,
    use_correlation=args.use_correlation,
    use_image=args.use_image,
    feature_extractor=args.feature_extractor
).to(device)

if os.path.exists(args.snapshot):
    print(f'Loading model from: {args.snapshot}')
    model.load_state_dict(torch.load(args.snapshot, map_location=device))
model.eval()

# **加载数据集**
test_transform = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.ToTensor()
])
test_dataset = Dataset([args.data_test], [args.heatmaps_test], [args.ann_path_test], [args.rois_path_test],
                        img_transform=test_transform, dataset=args.dataset)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

# **自动获取骨龄范围**
bone_ages_list = [sample[1].item() for sample in test_dataset]  # 取出所有骨龄值
bone_ages_series = pd.Series(bone_ages_list)
num_bins = 4  # 设定分成 4 段（可以调整）

# **自动分段骨龄**
bone_ages_series, bins = pd.qcut(bone_ages_series, num_bins, retbins=True, labels=False)
age_labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]

# **动态确定启用的特征**
enabled_features = []
if args.use_image: enabled_features.append("x")
if args.gender_multiplier: enabled_features.append("y")
if args.chronological_age: enabled_features.append("z")
if args.use_gut_microbiome: enabled_features.append("gut")
if args.use_pe_performance: enabled_features.append("pe")
if args.use_correlation: enabled_features.append("cor")

# **初始化贡献矩阵**
contribution_matrix = {feature: {age_label: 0.0 for age_label in age_labels} for feature in enabled_features}
age_counts = {age_label: 0 for age_label in age_labels}  # 记录每个年龄段的数据量

# **计算贡献度**
def compute_feature_contributions(model, dataloader, device, enabled_features):
    """
    计算不同年龄分段的模态贡献矩阵，并绘制热力图。

    参数:
        model: 训练好的 SIMBA 模型
        dataloader: DataLoader，加载验证集数据
        device: 运行设备 ('cuda' 或 'cpu')
        enabled_features: 由 `args` 传入的参与贡献计算的特征列表
    """
    # ✅ **初始化贡献矩阵**
    contribution_matrix = {feature: {age_label: 0.0 for age_label in age_labels} for feature in enabled_features}
    age_counts = {age_label: 0 for age_label in age_labels}  # 记录每个年龄段的数据量
    
    for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        # **完整加载输入**
        inputs, bone_ages, gender, chronological_age, p_id, gut, pe, cor = batch
        bone_ages = bone_ages.item()  # 取出骨龄值
        feature_dict = {
            "x": inputs.to(device),
            "y": gender.to(device),
            "z": chronological_age.to(device),
            "gut": gut.to(device),
            "pe": pe.to(device),
            "cor": cor.to(device)
        }

        # **自动匹配所属年龄段**
        age_index = pd.cut([bone_ages], bins=bins, labels=age_labels)[0]
        if pd.isna(age_index): continue  # 超出范围的跳过
        age_counts[age_index] += 1

        # ✅ **计算基准预测**
        with torch.no_grad():
            baseline_output = model(**feature_dict)

        # ✅ **仅对 `args` 传入的特征计算贡献**
        for feature_name in enabled_features:
            with torch.no_grad():
                masked_feature_dict = feature_dict.copy()
                masked_feature_dict[feature_name] = torch.zeros_like(masked_feature_dict[feature_name]).to(device)  # 置零

                new_output = model(**masked_feature_dict)
                contribution_matrix[feature_name][age_index] += torch.abs(baseline_output - new_output).mean().item()

    # # **归一化贡献度**
    # for feature in enabled_features:
    #     for age_group in age_labels:
    #         if age_counts[age_group] > 0:
    #             contribution_matrix[feature][age_group] /= age_counts[age_group]
    # 归一化贡献度，使每个年龄段的总贡献度为 100%
    for age_group in age_labels:
        total_contribution = sum(contribution_matrix[feature][age_group] for feature in enabled_features)
        if total_contribution > 0:  # 避免除零错误
            for feature in enabled_features:
                contribution_matrix[feature][age_group] = (contribution_matrix[feature][age_group] / total_contribution) * 100  # 归一化到 100%

    # **修改 key 名称映射**
    key_mapping = {"x": "image", "y": "gender", "z": "age", "cor": "correlation_features", 'gut':"nutrition", 'pe':"pe"}
    contribution_matrix = {key_mapping.get(k, k): v for k, v in contribution_matrix.items()}
    return contribution_matrix

# **绘制热力图**
def plot_contribution_matrix(contribution_matrix, save_folder):
    """
    绘制模态贡献热力图，并保存到指定路径。

    参数:
        contribution_matrix: 计算得到的贡献矩阵
        save_folder: 可视化图片保存目录
    """
    os.makedirs(save_folder, exist_ok=True)
    
    df = pd.DataFrame(contribution_matrix)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap="Blues", fmt=".4f", linewidths=0.5, cbar=True)
    plt.xlabel("Modality")
    plt.ylabel("Bone Age Group")
    plt.title("Feature Contribution by Bone Age Group")
    
    save_path = os.path.join(save_folder, "feature_contribution_heatmap.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 贡献度热力图已保存至: {save_path}")

# **运行计算**
print("🚀 开始计算年龄分段的特征贡献矩阵...")
contribution_matrix = compute_feature_contributions(model, test_loader, device, enabled_features)
plot_contribution_matrix(contribution_matrix, args.save_folder)
print("🎯 任务完成！")










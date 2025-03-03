import torch
import numpy as np
import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import inspect
from bone_age_assessment.BA.simba.models.simba_bk import SIMBA
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

# **动态多模态选择**
parser.add_argument('--relative-age', default=False, action='store_true')
parser.add_argument('--chronological-age', default=False, action='store_true')
parser.add_argument('--gender-multiplier', default=False, action='store_true')
parser.add_argument('--use-image', action='store_true')
parser.add_argument('--use-gut-microbiome', action='store_true')
parser.add_argument('--use-pe-performance', action='store_true')
parser.add_argument('--use-correlation', action='store_true')

args = parser.parse_args()

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# **加载模型**
model = SIMBA(
    chronological_age=args.chronological_age,
    gender_multiplier=args.gender_multiplier,
    use_gut_microbiome=args.use_gut_microbiome,
    use_pe_performance=args.use_pe_performance,
    use_correlation=args.use_correlation,
    use_image=args.use_image
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
bone_ages_list = [sample[1].item() for sample in test_dataset]
bone_ages_series = pd.Series(bone_ages_list)
num_bins = 4  # 分 4 段

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
age_counts = {age_label: 0 for age_label in age_labels}

# **计算 SHAP 贡献度**
def compute_shap_contributions(model, dataloader, device, enabled_features):
    """
    计算 SHAP 值，并绘制年龄分段的模态贡献矩阵。

    参数:
        model: 训练好的 SIMBA 模型
        dataloader: DataLoader，加载验证集数据
        device: 运行设备 ('cuda' 或 'cpu')
        enabled_features: 由 `args` 传入的参与贡献计算的特征列表
    """
    model_forward_params = inspect.signature(model.forward).parameters
    required_features = list(model_forward_params.keys())  # 获取 forward() 需要的参数名

    # **提取 `dataloader` 里第一个 batch，用于初始化 SHAP**
    sample_batch = next(iter(dataloader))
    inputs, bone_ages, gender, chronological_age, p_id, gut, pe, cor = sample_batch

    feature_dict = {
        "x": inputs.to(device),
        "y": gender.to(device),
        "z": chronological_age.to(device),
        "gut": gut.to(device),
        "pe": pe.to(device),
        "cor": cor.to(device)
    }

    # 🚀 确保 feature_dict 按照 forward() 需要的顺序排列
    # 🚀 确保 feature_list 里的张量可以计算梯度
    feature_list = [
        feature_dict["x"].requires_grad_(),   # 图像特征
        feature_dict["y"].requires_grad_(),   # 性别
        feature_dict["z"].requires_grad_(),   # 年龄
        feature_dict["gut"].requires_grad_(), # 肠道微生物
        feature_dict["pe"].requires_grad_(),  # 体能测试
        feature_dict["cor"].requires_grad_()  # 相关性特征
    ]

    # ✅ 传入 SHAP 解释器
    explainer = shap.GradientExplainer(model, feature_list)



    for batch in tqdm(dataloader, total=len(dataloader)):
        inputs, bone_ages, gender, chronological_age, p_id, gut, pe, cor = batch
        bone_ages = bone_ages.item()
        feature_dict = {
            "x": inputs.to(device),
            "y": gender.to(device),
            "z": chronological_age.to(device),
            "gut": gut.to(device),
            "pe": pe.to(device),
            "cor": cor.to(device)
        }
        feature_dict = {k: v for k, v in feature_dict.items() if k in required_features}
        # 🚀 确保 feature_dict 按照 forward() 需要的顺序排列
        # 🚀 确保 feature_list 里的张量可以计算梯度
        feature_list = [
            feature_dict["x"].requires_grad_(),   # 图像特征
            feature_dict["y"].requires_grad_(),   # 性别
            feature_dict["z"].requires_grad_(),   # 年龄
            feature_dict["gut"].requires_grad_(), # 肠道微生物
            feature_dict["pe"].requires_grad_(),  # 体能测试
            feature_dict["cor"].requires_grad_()  # 相关性特征
        ]

        age_index = pd.cut([bone_ages], bins=bins, labels=age_labels)[0]
        if pd.isna(age_index): continue
        age_counts[age_index] += 1

        with torch.no_grad():
            shap_values = explainer.shap_values(feature_list)

        for idx, feature_name in enumerate(enabled_features):
            contribution_matrix[feature_name][age_index] += np.mean(np.abs(shap_values[idx]))

    for feature in enabled_features:
        for age_group in age_labels:
            if age_counts[age_group] > 0:
                contribution_matrix[feature][age_group] /= age_counts[age_group]

    return contribution_matrix

# **绘制贡献热力图**
def plot_contribution_matrix(contribution_matrix, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    
    df = pd.DataFrame(contribution_matrix)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap="Blues", fmt=".4f", linewidths=0.5, cbar=True)
    plt.xlabel("Bone Age Group")
    plt.ylabel("Modality")
    plt.title("SHAP Feature Contribution by Bone Age Group")
    
    save_path = os.path.join(save_folder, "shap_feature_contribution_heatmap.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ SHAP 贡献度热力图已保存至: {save_path}")

# **运行计算**
print("🚀 开始计算 SHAP 贡献矩阵...")
contribution_matrix = compute_shap_contributions(model, test_loader, device, enabled_features)
plot_contribution_matrix(contribution_matrix, args.save_folder)
print("🎯 任务完成！")

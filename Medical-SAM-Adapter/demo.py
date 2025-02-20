import pandas as pd
from sklearn.model_selection import train_test_split

# 读取原始的annotation.csv文件
df = pd.read_csv('/private/workspace/cyt/bone_age_assessment/data/RSNA/segmentation/annotations.csv')

# 假设原始csv中没有标题行，若有标题行可以跳过此步骤
# df = pd.read_csv('annotation.csv', header=0)

# 随机划分数据集，80%用于训练，20%用于验证
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 保存为annotation_train.csv和annotation_test.csv
train_df.to_csv('/private/workspace/cyt/bone_age_assessment/data/RSNA/segmentation/annotations_train.csv', index=False)
test_df.to_csv('/private/workspace/cyt/bone_age_assessment/data/RSNA/segmentation/annotations_test.csv', index=False)

print("训练集和验证集已经分割并保存为 annotation_train.csv 和 annotation_test.csv")
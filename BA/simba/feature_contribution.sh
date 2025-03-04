#!/bin/bash

# 设定批量大小、工作线程数、GPU 设备
BATCH_SIZE=1
NUM_WORKERS=4
NUM_GPUS=1
GPUS=0
FEATURE_EXTRACTOR=resnet

# 路径设置
SSD_LOCATION='/private/workspace/cyt/bone_age_assessment/BA/simba'
DATASET="KG"
EXPERIMENT_NAME="best_experiment/"$DATASET/with_gender_c_age_pe_gut_cor_relative_resnet # with_gender_c_age_relative_resnet, with_gut_pe_abs_resnet, with_pe_gut_cor_abs_resnet, with_gender_c_age_pe_gut_cor_relative_resnet


DATA_TEST='/private/workspace/cyt/bone_age_assessment/data/data_yuwei/val_clean'  # 测试图像路径
HEATMAPS_TEST=$SSD_LOCATION"/experiments/"$EXPERIMENT_NAME"/HEATMAPS_TEST" # 测试热力图路径 (自动创建)
ANN_PATH_TEST='/private/workspace/cyt/bone_age_assessment/data/data_yuwei/annotations/val_ann.csv'  # CSV 标注路径
ROIS_PATH_TEST='/private/workspace/cyt/bone_age_assessment/data/RSNA/annotations/RSNA_Anatomical_ROIs_Validation.json' # ROI JSON 标注路径

# 结果保存路径
SAVE_FOLDER=$SSD_LOCATION"/experiments/"$EXPERIMENT_NAME
CONTRIBUTION_FOLDER=$SAVE_FOLDER"/feature_contribution"

# 创建目录（如果不存在）
mkdir -p $HEATMAPS_TEST
mkdir -p $SAVE_FOLDER
mkdir -p $CONTRIBUTION_FOLDER

# 模型权重路径
SNAPSHOT=$SAVE_FOLDER"/boneage_bonet_weights.pth"

# 输出文件
SAVE_FILE="validation_bestmodel.csv"

# # image + c_age
# CUDA_VISIBLE_DEVICES=$GPUS python feature_contribution_mask.py \
#     --data-test $DATA_TEST \
#     --ann-path-test $ANN_PATH_TEST \
#     --rois-path-test $ROIS_PATH_TEST \
#     --heatmaps-test $HEATMAPS_TEST \
#     --batch-size $BATCH_SIZE \
#     --gpu $GPUS \
#     --save-folder $CONTRIBUTION_FOLDER \
#     --snapshot $SNAPSHOT \
#     --dataset $DATASET \
#     --workers $NUM_WORKERS \
#     --use-image \
#     --chronological-age \
#     --relative-age \
#     --feature-extractor $FEATURE_EXTRACTOR

# # image + pe
# CUDA_VISIBLE_DEVICES=$GPUS python feature_contribution_mask.py \
#     --data-test $DATA_TEST \
#     --ann-path-test $ANN_PATH_TEST \
#     --rois-path-test $ROIS_PATH_TEST \
#     --heatmaps-test $HEATMAPS_TEST \
#     --batch-size $BATCH_SIZE \
#     --gpu $GPUS \
#     --save-folder $CONTRIBUTION_FOLDER \
#     --snapshot $SNAPSHOT \
#     --dataset $DATASET \
#     --workers $NUM_WORKERS \
#     --use-image \
#     --use-pe-performance \
#     --feature-extractor $FEATURE_EXTRACTOR

# # image + gut
# CUDA_VISIBLE_DEVICES=$GPUS python feature_contribution_mask.py \
#     --data-test $DATA_TEST \
#     --ann-path-test $ANN_PATH_TEST \
#     --rois-path-test $ROIS_PATH_TEST \
#     --heatmaps-test $HEATMAPS_TEST \
#     --batch-size $BATCH_SIZE \
#     --gpu $GPUS \
#     --save-folder $CONTRIBUTION_FOLDER \
#     --snapshot $SNAPSHOT \
#     --dataset $DATASET \
#     --workers $NUM_WORKERS \
#     --use-image \
#     --use-gut-microbiome \
#     --feature-extractor $FEATURE_EXTRACTOR

# # image + cor
# CUDA_VISIBLE_DEVICES=$GPUS python feature_contribution_mask.py \
#     --data-test $DATA_TEST \
#     --ann-path-test $ANN_PATH_TEST \
#     --rois-path-test $ROIS_PATH_TEST \
#     --heatmaps-test $HEATMAPS_TEST \
#     --batch-size $BATCH_SIZE \
#     --gpu $GPUS \
#     --save-folder $CONTRIBUTION_FOLDER \
#     --snapshot $SNAPSHOT \
#     --dataset $DATASET \
#     --workers $NUM_WORKERS \
#     --use-image \
#     --use-correlation \
#     # --use-gender \
#     --feature-extractor $FEATURE_EXTRACTOR

# # image + c_age + gender
# CUDA_VISIBLE_DEVICES=$GPUS python feature_contribution_mask.py \
#     --data-test $DATA_TEST \
#     --ann-path-test $ANN_PATH_TEST \
#     --rois-path-test $ROIS_PATH_TEST \
#     --heatmaps-test $HEATMAPS_TEST \
#     --batch-size $BATCH_SIZE \
#     --gpu $GPUS \
#     --save-folder $CONTRIBUTION_FOLDER \
#     --snapshot $SNAPSHOT \
#     --dataset $DATASET \
#     --workers $NUM_WORKERS \
#     --use-image \
#     --chronological-age \
#     --relative-age \
#     --use-gender \
#     --gender-multiplier \
#     --feature-extractor $FEATURE_EXTRACTOR

# # image + gut + pe
# CUDA_VISIBLE_DEVICES=$GPUS python feature_contribution_mask.py \
#     --data-test $DATA_TEST \
#     --ann-path-test $ANN_PATH_TEST \
#     --rois-path-test $ROIS_PATH_TEST \
#     --heatmaps-test $HEATMAPS_TEST \
#     --batch-size $BATCH_SIZE \
#     --gpu $GPUS \
#     --save-folder $CONTRIBUTION_FOLDER \
#     --snapshot $SNAPSHOT \
#     --dataset $DATASET \
#     --workers $NUM_WORKERS \
#     --use-image \
#     --use-gut-microbiome \
#     --use-pe-performance \
#     --feature-extractor $FEATURE_EXTRACTOR

# # image + cor + gut + pe
# CUDA_VISIBLE_DEVICES=$GPUS python feature_contribution_mask.py \
#     --data-test $DATA_TEST \
#     --ann-path-test $ANN_PATH_TEST \
#     --rois-path-test $ROIS_PATH_TEST \
#     --heatmaps-test $HEATMAPS_TEST \
#     --batch-size $BATCH_SIZE \
#     --gpu $GPUS \
#     --save-folder $CONTRIBUTION_FOLDER \
#     --snapshot $SNAPSHOT \
#     --dataset $DATASET \
#     --workers $NUM_WORKERS \
#     --use-image \
#     --use-gut-microbiome \
#     --use-pe-performance \
#     --use-correlation \
#     --feature-extractor $FEATURE_EXTRACTOR

# image + cor + gut + pe
CUDA_VISIBLE_DEVICES=$GPUS python feature_contribution_mask.py \
    --data-test $DATA_TEST \
    --ann-path-test $ANN_PATH_TEST \
    --rois-path-test $ROIS_PATH_TEST \
    --heatmaps-test $HEATMAPS_TEST \
    --batch-size $BATCH_SIZE \
    --gpu $GPUS \
    --save-folder $CONTRIBUTION_FOLDER \
    --snapshot $SNAPSHOT \
    --dataset $DATASET \
    --workers $NUM_WORKERS \
    --use-image \
    --use-gut-microbiome \
    --use-pe-performance \
    --use-correlation \
    --chronological-age \
    --relative-age \
    --use-gender \
    --gender-multiplier \
    --feature-extractor $FEATURE_EXTRACTOR

echo "Testing completed. Results saved in $SAVE_FOLDER/$SAVE_FILE"
echo "feature contributions visualizations saved in $CONTRIBUTION_FOLDER"

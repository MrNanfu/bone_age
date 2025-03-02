#!/bin/bash

# 设定批量大小、工作线程数、GPU 设备
BATCH_SIZE=1
NUM_WORKERS=4
NUM_GPUS=1
GPUS=0

# 路径设置
SSD_LOCATION='/private/workspace/cyt/bone_age_assessment/BA/simba'
DATASET="KG"
EXPERIMENT_NAME="best_experiment/"$DATASET/with_only_image_seg_abs # with_only_image_seg_abs,  with_gender_c_age_cor_seg


DATA_TEST='/private/workspace/cyt/bone_age_assessment/data/data_yuwei/val_clean'  # 测试图像路径
HEATMAPS_TEST=$SSD_LOCATION"/experiments/"$EXPERIMENT_NAME"/HEATMAPS_TEST" # 测试热力图路径 (自动创建)
ANN_PATH_TEST='/private/workspace/cyt/bone_age_assessment/data/data_yuwei/annotations/val_ann.csv'  # CSV 标注路径
ROIS_PATH_TEST='/private/workspace/cyt/bone_age_assessment/data/RSNA/annotations/RSNA_Anatomical_ROIs_Validation.json' # ROI JSON 标注路径

# 结果保存路径
SAVE_FOLDER=$SSD_LOCATION"/experiments/"$EXPERIMENT_NAME
GRADCAM_FOLDER=$SAVE_FOLDER"/gradcam"

# 创建目录（如果不存在）
mkdir -p $HEATMAPS_TEST
mkdir -p $SAVE_FOLDER
mkdir -p $GRADCAM_FOLDER

# 模型权重路径
SNAPSHOT=$SAVE_FOLDER"/boneage_bonet_weights.pth"

# 输出文件
SAVE_FILE="validation_bestmodel.csv"

# 运行骨龄预测测试并生成 Grad-CAM 可视化
# CUDA_VISIBLE_DEVICES=$GPUS python visualize.py \
#     --data-test $DATA_TEST \
#     --ann-path-test $ANN_PATH_TEST \
#     --rois-path-test $ROIS_PATH_TEST \
#     --heatmaps-test $HEATMAPS_TEST \
#     --batch-size $BATCH_SIZE \
#     --gpu $GPUS \
#     --save-folder $SAVE_FOLDER \
#     --save-file $SAVE_FILE \
#     --snapshot $SNAPSHOT \
#     --dataset $DATASET \
#     --workers $NUM_WORKERS \
#     --relative-age \
#     --chronological-age \
#     --gender-multiplie \
#     --use-correlation \
#     --use-image

    CUDA_VISIBLE_DEVICES=$GPUS python visualize.py \
    --data-test $DATA_TEST \
    --ann-path-test $ANN_PATH_TEST \
    --rois-path-test $ROIS_PATH_TEST \
    --heatmaps-test $HEATMAPS_TEST \
    --batch-size $BATCH_SIZE \
    --gpu $GPUS \
    --save-folder $SAVE_FOLDER \
    --save-file $SAVE_FILE \
    --snapshot $SNAPSHOT \
    --dataset $DATASET \
    --workers $NUM_WORKERS \
    --use-image

echo "Testing completed. Results saved in $SAVE_FOLDER/$SAVE_FILE"
echo "Grad-CAM visualizations saved in $GRADCAM_FOLDER"

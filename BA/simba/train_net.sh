#!/bin/bash
START_EPOCH=0
NUM_EPOCHS=150
LR=0.0001
PATIENCE=2
BATCH_SIZE=20
NUM_WORKERS=4
NUM_GPUS=1
GPUS=0

SSD_LOCATION='/private/workspace/cyt/bone_age_assessment/BA/simba'
DATASET="KG"
EXPERIMENT_NAME="best_experiment/"$DATASET/with_only_image_seg_abs # with_gender_c_age
SAVE_FOLDER=$SSD_LOCATION"/experiments/"$EXPERIMENT_NAME

DATA_TRAIN='/private/workspace/cyt/bone_age_assessment/data/data_yuwei/train_clean' #Path to  images
HEATMAPS_TRAIN=$SAVE_FOLDER"/HEATMAPS_TRAIN" #Path to heatmaps (Will be created automatically)
ANN_PATH_TRAIN='/private/workspace/cyt/bone_age_assessment/data/data_yuwei/annotations/train_ann.csv' #Path to csv annotations
ROIS_PATH_TRAIN='/private/workspace/cyt/bone_age_assessment/data/RSNA/annotations/RSNA_Anatomical_ROIs_Training.json' #Path to json annotations of ROIs

mkdir -p $HEATMAPS_TRAIN

DATA_VAL='/private/workspace/cyt/bone_age_assessment/data/data_yuwei/val_clean' #Path to  images
HEATMAPS_VAL=$SAVE_FOLDER"/HEATMAPS_VAL" #Path to heatmaps (Will be created automatically)
ANN_PATH_VAL='/private/workspace/cyt/bone_age_assessment/data/data_yuwei/annotations/val_ann.csv' #Path to csv annotations
ROIS_PATH_VAL='/private/workspace/cyt/bone_age_assessment/data/RSNA/annotations/RSNA_Anatomical_ROIs_Validation.json' #Path to json annotations of ROIs

mkdir -p $HEATMAPS_VAL



mkdir -p $SAVE_FOLDER

SNAPSHOT=$SAVE_FOLDER"/boneage_bonet_snapshot.pth"
OPTIM_SNAPSHOT=$SAVE_FOLDER"/boneage_bonet_optim.pth"

# only image
CUDA_VISIBLE_DEVICES=$GPUS python -m train --data-train $DATA_TRAIN --heatmaps-train $HEATMAPS_TRAIN --ann-path-train $ANN_PATH_TRAIN --rois-path-train $ROIS_PATH_TRAIN --data-val $DATA_VAL --heatmaps-val $HEATMAPS_VAL --ann-path-val $ANN_PATH_VAL --rois-path-val $ROIS_PATH_VAL --batch-size $BATCH_SIZE --start-epoch $START_EPOCH --epochs $NUM_EPOCHS --lr $LR --patience $PATIENCE --gpu $GPUS --save-folder $SAVE_FOLDER --dataset $DATASET --workers $NUM_WORKERS --start-epoch $START_EPOCH --snapshot $SNAPSHOT --optim-snapshot $OPTIM_SNAPSHOT --trainval --eval-first

# image + gender + chronological_age + correlation_features
# CUDA_VISIBLE_DEVICES=$GPUS python -m train --data-train $DATA_TRAIN --heatmaps-train $HEATMAPS_TRAIN --ann-path-train $ANN_PATH_TRAIN --rois-path-train $ROIS_PATH_TRAIN --data-val $DATA_VAL --heatmaps-val $HEATMAPS_VAL --ann-path-val $ANN_PATH_VAL --rois-path-val $ROIS_PATH_VAL --batch-size $BATCH_SIZE --start-epoch $START_EPOCH --epochs $NUM_EPOCHS --lr $LR --patience $PATIENCE --gpu $GPUS --save-folder $SAVE_FOLDER --dataset $DATASET --workers $NUM_WORKERS --start-epoch $START_EPOCH --snapshot $SNAPSHOT --optim-snapshot $OPTIM_SNAPSHOT --trainval --eval-first --relative-age --chronological-age --gender-multiplier --use-correlation

# image + correlation_features
# CUDA_VISIBLE_DEVICES=$GPUS python -m train --data-train $DATA_TRAIN --heatmaps-train $HEATMAPS_TRAIN --ann-path-train $ANN_PATH_TRAIN --rois-path-train $ROIS_PATH_TRAIN --data-val $DATA_VAL --heatmaps-val $HEATMAPS_VAL --ann-path-val $ANN_PATH_VAL --rois-path-val $ROIS_PATH_VAL --batch-size $BATCH_SIZE --start-epoch $START_EPOCH --epochs $NUM_EPOCHS --lr $LR --patience $PATIENCE --gpu $GPUS --save-folder $SAVE_FOLDER --dataset $DATASET --workers $NUM_WORKERS --start-epoch $START_EPOCH --snapshot $SNAPSHOT --optim-snapshot $OPTIM_SNAPSHOT --trainval --eval-first --relative-age --use-correlation

# image + gender +  chronological_age
# CUDA_VISIBLE_DEVICES=$GPUS python -m train --data-train $DATA_TRAIN --heatmaps-train $HEATMAPS_TRAIN --ann-path-train $ANN_PATH_TRAIN --rois-path-train $ROIS_PATH_TRAIN --data-val $DATA_VAL --heatmaps-val $HEATMAPS_VAL --ann-path-val $ANN_PATH_VAL --rois-path-val $ROIS_PATH_VAL --batch-size $BATCH_SIZE --start-epoch $START_EPOCH --epochs $NUM_EPOCHS --lr $LR --patience $PATIENCE --gpu $GPUS --save-folder $SAVE_FOLDER --dataset $DATASET --workers $NUM_WORKERS --start-epoch $START_EPOCH --snapshot $SNAPSHOT --optim-snapshot $OPTIM_SNAPSHOT --trainval --eval-first --relative-age --chronological-age --gender-multiplier


# CUDA_VISIBLE_DEVICES=$GPUS python -m train --data-train $DATA_TRAIN --heatmaps-train $HEATMAPS_TRAIN --ann-path-train $ANN_PATH_TRAIN --rois-path-train $ROIS_PATH_TRAIN --data-val $DATA_VAL --heatmaps-val $HEATMAPS_VAL --ann-path-val $ANN_PATH_VAL --rois-path-val $ROIS_PATH_VAL --batch-size $BATCH_SIZE --start-epoch $START_EPOCH --epochs $NUM_EPOCHS --lr $LR --patience $PATIENCE --gpu $GPUS --save-folder $SAVE_FOLDER --dataset $DATASET --workers $NUM_WORKERS --start-epoch $START_EPOCH --snapshot $SNAPSHOT --optim-snapshot $OPTIM_SNAPSHOT --trainval --eval-first --relative-age --chronological-age --gender-multiplier


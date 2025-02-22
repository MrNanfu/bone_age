#!/bin/bash
START_EPOCH=0
NUM_EPOCHS=150
LR=0.0001
PATIENCE=2
BATCH_SIZE=20
NUM_WORKERS=25
NUM_GPUS=1
GPUS=0

SSD_LOCATION='.'
DATASET="KG"
EXPERIMENT_NAME="best_experiment/"$DATASET/with_gut_pe
SAVE_FOLDER=$SSD_LOCATION"/experiments/"$EXPERIMENT_NAME

DATA_TRAIN='/private/workspace/cyt/bone_age_assessment/data/data_yuwei/train' #Path to  images
HEATMAPS_TRAIN=$SAVE_FOLDER"/HEATMAPS_TRAIN" #Path to heatmaps (Will be created automatically)
ANN_PATH_TRAIN='/private/workspace/cyt/bone_age_assessment/data/data_yuwei/annotations/train_ann.csv' #Path to csv annotations
ROIS_PATH_TRAIN='/private/workspace/cyt/bone_age_assessment/data/RSNA/annotations/RSNA_Anatomical_ROIs_Training.json' #Path to json annotations of ROIs

mkdir -p $HEATMAPS_TRAIN

DATA_VAL='/private/workspace/cyt/bone_age_assessment/data/data_yuwei/val' #Path to  images
HEATMAPS_VAL=$SAVE_FOLDER"/HEATMAPS_VAL" #Path to heatmaps (Will be created automatically)
ANN_PATH_VAL='/private/workspace/cyt/bone_age_assessment/data/data_yuwei/annotations/val_ann.csv' #Path to csv annotations
ROIS_PATH_VAL='/private/workspace/cyt/bone_age_assessment/data/RSNA/annotations/RSNA_Anatomical_ROIs_Validation.json' #Path to json annotations of ROIs

mkdir -p $HEATMAPS_VAL



mkdir -p $SAVE_FOLDER

SNAPSHOT=$SAVE_FOLDER"/boneage_bonet_snapshot.pth"
OPTIM_SNAPSHOT=$SAVE_FOLDER"/boneage_bonet_optim.pth"

CUDA_VISIBLE_DEVICES=$GPUS python -m train --data-train $DATA_TRAIN --heatmaps-train $HEATMAPS_TRAIN --ann-path-train $ANN_PATH_TRAIN --rois-path-train $ROIS_PATH_TRAIN --data-val $DATA_VAL --heatmaps-val $HEATMAPS_VAL --ann-path-val $ANN_PATH_VAL --rois-path-val $ROIS_PATH_VAL --batch-size $BATCH_SIZE --start-epoch $START_EPOCH --epochs $NUM_EPOCHS --lr $LR --patience $PATIENCE --gpu $GPUS --save-folder $SAVE_FOLDER --dataset $DATASET --workers $NUM_WORKERS --start-epoch $START_EPOCH --snapshot $SNAPSHOT --optim-snapshot $OPTIM_SNAPSHOT --trainval --eval-first --relative-age --chronological-age --gender-multiplier --use-gut-microbiome --use-pe-performance

# CUDA_VISIBLE_DEVICES=$GPUS python -m train --data-train $DATA_TRAIN --heatmaps-train $HEATMAPS_TRAIN --ann-path-train $ANN_PATH_TRAIN --rois-path-train $ROIS_PATH_TRAIN --data-val $DATA_VAL --heatmaps-val $HEATMAPS_VAL --ann-path-val $ANN_PATH_VAL --rois-path-val $ROIS_PATH_VAL --batch-size $BATCH_SIZE --start-epoch $START_EPOCH --epochs $NUM_EPOCHS --lr $LR --patience $PATIENCE --gpu $GPUS --save-folder $SAVE_FOLDER --dataset $DATASET --workers $NUM_WORKERS --start-epoch $START_EPOCH --snapshot $SNAPSHOT --optim-snapshot $OPTIM_SNAPSHOT --trainval --eval-first --relative-age --chronological-age --gender-multiplier

# python -m train --data-train $DATA_TRAIN --heatmaps-train $HEATMAPS_TRAIN --ann-path-train $ANN_PATH_TRAIN --rois-path-train $ROIS_PATH_TRAIN --data-val $DATA_VAL --heatmaps-val $HEATMAPS_VAL --ann-path-val $ANN_PATH_VAL --rois-path-val $ROIS_PATH_VAL --batch-size $BATCH_SIZE --start-epoch $START_EPOCH --epochs $NUM_EPOCHS --lr $LR --patience $PATIENCE --gpu $GPUS --save-folder $SAVE_FOLDER --dataset $DATASET --workers $NUM_WORKERS --start-epoch $START_EPOCH --cropped --snapshot $SNAPSHOT --optim-snapshot $OPTIM_SNAPSHOT --trainval --eval-first --relative-age --chronological-age --gender-multiplier #>> $SAVE_FOLDER"/log.txt"
# CUDA_VISIBLE_DEVICES=$GPUS mpirun --allow-run-as-root -np $NUM_GPUS -H localhost:$NUM_GPUS -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python -m train --data-train $DATA_TRAIN --heatmaps-train $HEATMAPS_TRAIN --ann-path-train $ANN_PATH_TRAIN --rois-path-train $ROIS_PATH_TRAIN --data-val $DATA_VAL --heatmaps-val $HEATMAPS_VAL --ann-path-val $ANN_PATH_VAL --rois-path-val $ROIS_PATH_VAL --batch-size $BATCH_SIZE --start-epoch $START_EPOCH --epochs $NUM_EPOCHS --lr $LR --patience $PATIENCE --gpu $GPUS --save-folder $SAVE_FOLDER --dataset $DATASET --workers $NUM_WORKERS --start-epoch $START_EPOCH --cropped --snapshot $SNAPSHOT --optim-snapshot $OPTIM_SNAPSHOT --trainval --eval-first --relative-age --chronological-age --gender-multiplier #>> $SAVE_FOLDER"/log.txt"
# CUDA_VISIBLE_DEVICES=$GPUS mpirun --allow-run-as-root -np $NUM_GPUS -H localhost:$NUM_GPUS -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x HOROVOD_CUDA_HOME=/usr/local/cuda-10.0 -mca pml ob1 -mca btl ^openib python -m train --data-train $DATA_TRAIN --heatmaps-train $HEATMAPS_TRAIN --ann-path-train $ANN_PATH_TRAIN --rois-path-train $ROIS_PATH_TRAIN --data-val $DATA_VAL --heatmaps-val $HEATMAPS_VAL --ann-path-val $ANN_PATH_VAL --rois-path-val $ROIS_PATH_VAL --batch-size $BATCH_SIZE --start-epoch $START_EPOCH --epochs $NUM_EPOCHS --lr $LR --patience $PATIENCE --gpu $GPUS --save-folder $SAVE_FOLDER --dataset $DATASET --workers $NUM_WORKERS --start-epoch $START_EPOCH --cropped --snapshot $SNAPSHOT --optim-snapshot $OPTIM_SNAPSHOT --trainval --eval-first --relative-age --chronological-age --gender-multiplier #>> $SAVE_FOLDER"/log.txt"

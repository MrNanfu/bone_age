#!/bin/bash
BATCH_SIZE=1
NUM_WORKERS=4
NUM_GPUS=1
GPUS=0

SSD_LOCATION='/private/workspace/cyt/bone_age_assessment/BA/simba'
DATASET="KG"
EXPERIMENT_NAME="best_experiment/"$DATASET/with_only_image

DATA_TEST='/private/workspace/cyt/bone_age_assessment/data/data_yuwei/val' #Path to test images
HEATMAPS_TEST=$SAVE_FOLDER"/HEATMAPS_TEST" #Path to test heatmaps (Will be created automatically)
ANN_PATH_TEST='/private/workspace/cyt/bone_age_assessment/data/data_yuwei/annotations/val_ann.csv' #Path to csv annotations
ROIS_PATH_TEST='/private/workspace/cyt/bone_age_assessment/data/RSNA/annotations/RSNA_Anatomical_ROIs_Validation.json' #Path to json annotations of ROIs
  
mkdir -p $HEATMAPS_TEST
  
SAVE_FOLDER=$SSD_LOCATION"/experiments/"$EXPERIMENT_NAME
 
mkdir -p $SAVE_FOLDER

SNAPSHOT=$SAVE_FOLDER"/boneage_bonet_weights.pth"
SAVE_FILE="validation_bestmodel.csv"


CUDA_VISIBLE_DEVICES=$GPUS python test.py --data-test $DATA_TEST --ann-path-test $ANN_PATH_TEST  --rois-path-test $ROIS_PATH_TEST --heatmaps-test $HEATMAPS_TEST --batch-size $BATCH_SIZE --gpu $GPUS --save-folder $SAVE_FOLDER --save-file $SAVE_FILE --snapshot $SNAPSHOT --dataset $DATASET --workers $NUM_WORKERS  --relative-age
# CUDA_VISIBLE_DEVICES=$GPUS python test.py --data-test $DATA_TEST --ann-path-test $ANN_PATH_TEST  --rois-path-test $ROIS_PATH_TEST --heatmaps-test $HEATMAPS_TEST --batch-size $BATCH_SIZE --gpu $GPUS --save-folder $SAVE_FOLDER --save-file $SAVE_FILE --snapshot $SNAPSHOT --dataset $DATASET --workers $NUM_WORKERS --chronological-age --relative-age --gender-multiplier --use-correlation

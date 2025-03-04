#!/bin/bash
BATCH_SIZE=1
NUM_WORKERS=2
NUM_GPUS=1
GPUS=0
FEATURE_EXTRACTOR=resnet

SSD_LOCATION='/private/workspace/cyt/bone_age_assessment/BA/simba'
DATASET="KG"
EXPERIMENT_NAME="best_experiment/"$DATASET/with_gut_pe_abs_resnet

DATA_TEST='/private/workspace/cyt/bone_age_assessment/data/data_yuwei/val_clean' #Path to test images
HEATMAPS_TEST=$SAVE_FOLDER"/HEATMAPS_TEST" #Path to test heatmaps (Will be created automatically)
ANN_PATH_TEST='/private/workspace/cyt/bone_age_assessment/data/data_yuwei/annotations/val_ann.csv' #Path to csv annotations
ROIS_PATH_TEST='/private/workspace/cyt/bone_age_assessment/data/RSNA/annotations/RSNA_Anatomical_ROIs_Validation.json' #Path to json annotations of ROIs
  
mkdir -p $HEATMAPS_TEST
  
SAVE_FOLDER=$SSD_LOCATION"/experiments/"$EXPERIMENT_NAME
 
mkdir -p $SAVE_FOLDER

SNAPSHOT=$SAVE_FOLDER"/boneage_bonet_weights.pth"
SAVE_FILE="validation_bestmodel.csv"

# image + gender
# CUDA_VISIBLE_DEVICES=$GPUS python test.py --data-test $DATA_TEST --ann-path-test $ANN_PATH_TEST  --rois-path-test $ROIS_PATH_TEST --heatmaps-test $HEATMAPS_TEST --batch-size $BATCH_SIZE --gpu $GPUS --save-folder $SAVE_FOLDER --save-file $SAVE_FILE --snapshot $SNAPSHOT --dataset $DATASET --workers $NUM_WORKERS  --feature-extractor $FEATURE_EXTRACTOR --gender-multiplier --use-image

# image + c_age
# CUDA_VISIBLE_DEVICES=$GPUS python test.py --data-test $DATA_TEST --ann-path-test $ANN_PATH_TEST  --rois-path-test $ROIS_PATH_TEST --heatmaps-test $HEATMAPS_TEST --batch-size $BATCH_SIZE --gpu $GPUS --save-folder $SAVE_FOLDER --save-file $SAVE_FILE --snapshot $SNAPSHOT --dataset $DATASET --workers $NUM_WORKERS  --feature-extractor $FEATURE_EXTRACTOR  --use-image --relative-age --chronological-age

# image + pe
# CUDA_VISIBLE_DEVICES=$GPUS python test.py --data-test $DATA_TEST --ann-path-test $ANN_PATH_TEST  --rois-path-test $ROIS_PATH_TEST --heatmaps-test $HEATMAPS_TEST --batch-size $BATCH_SIZE --gpu $GPUS --save-folder $SAVE_FOLDER --save-file $SAVE_FILE --snapshot $SNAPSHOT --dataset $DATASET --workers $NUM_WORKERS  --feature-extractor $FEATURE_EXTRACTOR  --use-image --use-pe-performance

# image + gut
# CUDA_VISIBLE_DEVICES=$GPUS python test.py --data-test $DATA_TEST --ann-path-test $ANN_PATH_TEST  --rois-path-test $ROIS_PATH_TEST --heatmaps-test $HEATMAPS_TEST --batch-size $BATCH_SIZE --gpu $GPUS --save-folder $SAVE_FOLDER --save-file $SAVE_FILE --snapshot $SNAPSHOT --dataset $DATASET --workers $NUM_WORKERS  --feature-extractor $FEATURE_EXTRACTOR  --use-image --use-gut-microbiome

# # image + cor
# CUDA_VISIBLE_DEVICES=$GPUS python test.py --data-test $DATA_TEST --ann-path-test $ANN_PATH_TEST  --rois-path-test $ROIS_PATH_TEST --heatmaps-test $HEATMAPS_TEST --batch-size $BATCH_SIZE --gpu $GPUS --save-folder $SAVE_FOLDER --save-file $SAVE_FILE --snapshot $SNAPSHOT --dataset $DATASET --workers $NUM_WORKERS  --feature-extractor $FEATURE_EXTRACTOR  --use-image --use-correlation

# image + gedner + c_age
# CUDA_VISIBLE_DEVICES=$GPUS python test.py --data-test $DATA_TEST --ann-path-test $ANN_PATH_TEST  --rois-path-test $ROIS_PATH_TEST --heatmaps-test $HEATMAPS_TEST --batch-size $BATCH_SIZE --gpu $GPUS --save-folder $SAVE_FOLDER --save-file $SAVE_FILE --snapshot $SNAPSHOT --dataset $DATASET --workers $NUM_WORKERS  --feature-extractor $FEATURE_EXTRACTOR --use-gender --gender-multiplier --use-image --relative-age --chronological-age

# image + gut +pe
CUDA_VISIBLE_DEVICES=$GPUS python test.py --data-test $DATA_TEST --ann-path-test $ANN_PATH_TEST  --rois-path-test $ROIS_PATH_TEST --heatmaps-test $HEATMAPS_TEST --batch-size $BATCH_SIZE --gpu $GPUS --save-folder $SAVE_FOLDER --save-file $SAVE_FILE --snapshot $SNAPSHOT --dataset $DATASET --workers $NUM_WORKERS  --feature-extractor $FEATURE_EXTRACTOR  --use-image --use-gut-microbiome --use-pe-performance

# image + gut + pe +cor
# CUDA_VISIBLE_DEVICES=$GPUS python test.py --data-test $DATA_TEST --ann-path-test $ANN_PATH_TEST  --rois-path-test $ROIS_PATH_TEST --heatmaps-test $HEATMAPS_TEST --batch-size $BATCH_SIZE --gpu $GPUS --save-folder $SAVE_FOLDER --save-file $SAVE_FILE --snapshot $SNAPSHOT --dataset $DATASET --workers $NUM_WORKERS  --feature-extractor $FEATURE_EXTRACTOR  --use-image --use-gut-microbiome --use-pe-performance --use-correlation

# image + gut + pe + cor + gedner + c_age
# CUDA_VISIBLE_DEVICES=$GPUS python test.py --data-test $DATA_TEST --ann-path-test $ANN_PATH_TEST  --rois-path-test $ROIS_PATH_TEST --heatmaps-test $HEATMAPS_TEST --batch-size $BATCH_SIZE --gpu $GPUS --save-folder $SAVE_FOLDER --save-file $SAVE_FILE --snapshot $SNAPSHOT --dataset $DATASET --workers $NUM_WORKERS  --feature-extractor $FEATURE_EXTRACTOR  --use-image --use-gut-microbiome --use-pe-performance --use-correlation --gender-multiplier --use-gender --relative-age --chronological-age
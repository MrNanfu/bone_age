# Set the path to save images
OUTPUT_DIR='/private/workspace/cyt/bone_age_assessment/MAE-pytorch/output/visualization'
# path to image for visualization
IMAGE_PATH='/private/workspace/cyt/bone_age_assessment/data/data_yuwei/train_clean/127.png'
# IMAGE_PATH='/private/workspace/cyt/bone_age_assessment/MAE-pytorch/output/visualization/1029.png'
# INPUT_SIZE=224

# path to pretrain model
MODEL_PATH='/private/workspace/cyt/bone_age_assessment/MAE-pytorch/output/pretrain_mae_base_patch16_224/checkpoint-399.pth'

# Now, it only supports pretrained models with normalized pixel targets
python run_mae_vis.py --img_path ${IMAGE_PATH} --save_path ${OUTPUT_DIR} --model_path ${MODEL_PATH}
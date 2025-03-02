# Set the path to save images
OUTPUT_DIR='output/'
# path to image for visualization
IMAGE_PATH='/private/workspace/cyt/bone_age_assessment/data/data_yuwei/train_clean/102.png'
# path to pretrain model
MODEL_PATH='/path/to/pretrain/checkpoint.pth'

# Now, it only supports pretrained models with normalized pixel targets
python run_mae_vis.py ${IMAGE_PATH} ${OUTPUT_DIR} ${MODEL_PATH}
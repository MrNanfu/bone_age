# Set the path to save checkpoints
OUTPUT_DIR='output/pretrain_mae_base_patch16_224'
# path to imagenet-1k train set
DATA_PATH='/private/workspace/cyt/bone_age_assessment/data/RSNA/all'
SAVE_SEQ=1


# batch_size can be adjusted according to the graphics card
python  run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_ratio 0.75 \
        --model pretrain_mae_base_patch16_224 \
        --batch_size 256 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 10 \
        --epochs 400 \
        --output_dir ${OUTPUT_DIR} \
        --save_ckpt_freq ${SAVE_SEQ}
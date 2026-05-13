export CUDA_VISIBLE_DEVICES=0,1,2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

###################################
# User configuration
###################################

# Activate the diffusion_planner conda environment before running this script.
TRAIN_SET_PATH="/root/autodl-tmp/Diffusion-Planner/npzdata_50000"
TRAIN_SET_LIST_PATH="/root/autodl-tmp/Diffusion-Planner/diffusion_planner_training.json"

# These vocab files must match the tokenization used for the npz dataset.
EGO_VOCAB_PATH="/root/autodl-tmp/Diffusion-Planner/npz2token_vocab_dataset/ego_vocab_1024.npz"
NBR_VOCAB_PATH="/root/autodl-tmp/Diffusion-Planner/npz2token_vocab_dataset/nbr_vocab_1024.npz"

SAVE_DIR="/root/autodl-tmp/Diffusion-Planner/yzy_output"
EXP_NAME="training_hybrid_continuous_v1"

NPROC_PER_NODE=3
MASTER_PORT=22323

###################################
# Training arguments
###################################

TRAIN_ARGS=(
    --train_set "$TRAIN_SET_PATH"
    --train_set_list "$TRAIN_SET_LIST_PATH"
    --vocab_path "$EGO_VOCAB_PATH"
    --nbr_vocab_path "$NBR_VOCAB_PATH"

    --save_dir "$SAVE_DIR"
    --name "$EXP_NAME"
    --save_utd 40

    --use_data_augment True

    --batch_size 576
    --num_workers 16
    --train_epochs 1000

    # Token latent diffusion loss.
    --alpha_planning_loss 1.0

    # Final trajectory for sim comes from the continuous head.
    --use_continuous_head True
    --trajectory_output_mode continuous_head
    --continuous_head_hidden_dim 512
    --lambda_continuous_traj 1.0

    # Token classification is auxiliary supervision only.
    --learnable_token_emb False
    --use_token_classifier True
    --token_selection_mode classifier
    --lambda_token_cls_ce 0.2

    --ddp True
)

torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="$MASTER_PORT" \
    train_predictor.py \
    "${TRAIN_ARGS[@]}"

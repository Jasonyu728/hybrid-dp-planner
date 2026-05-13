export CUDA_VISIBLE_DEVICES=2,3
export HYDRA_FULL_ERROR=1

###################################
# User Configuration Section
###################################
# Set environment variables
export NUPLAN_DEVKIT_ROOT="/root/autodl-tmp/Diffusion-Planner/nuplan-devkit"
export NUPLAN_DATA_ROOT="/root/autodl-tmp/Diffusion-Planner/nuplan-devkit/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/root/autodl-tmp/Diffusion-Planner/nuplan-devkit/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="/root/autodl-tmp/Diffusion-Planner/nuplan-devkit/nuplan/dataset/exp"

# Dataset split to use
# Options:
#   - "test14-random"
#   - "test14-hard"
#   - "val14"
SPLIT="val14-collision"

# Challenge type
# Options:
#   - "closed_loop_nonreactive_agents"   (NR: 周围车辆不对 ego 做出反应)
#   - "closed_loop_reactive_agents"      (R:  周围车辆会对 ego 做出反应)
CHALLENGE="closed_loop_nonreactive_agents"
TOKEN_DECODE_MODE="hard"
TOKEN_DECODE_TEMPERATURE="1.0"
###################################


BRANCH_NAME=token_guidance
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$SCRIPT_DIR:$NUPLAN_DEVKIT_ROOT:$PYTHONPATH"

###################################
# Checkpoint Configuration
# 每次评估前修改 CKPT_NAME 指向对应的训练结果目录
CKPT_NAME="training_divide_0428/2026-04-28-15:07:06"
CKPT_DIR="$SCRIPT_DIR/yzy_output/training_log/$CKPT_NAME"
###################################

ARGS_FILE=$CKPT_DIR/args.json
CKPT_FILE=$CKPT_DIR/latest.pth

if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan"
else
    SCENARIO_BUILDER="nuplan_challenge"
fi
echo "Processing $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=diffusion_planner_token_guidance

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.diffusion_planner.config.args_file=$ARGS_FILE \
    planner.diffusion_planner.ckpt_path=$CKPT_FILE \
    scenario_builder=$SCENARIO_BUILDER \
    scenario_filter=$SPLIT \
    experiment_uid=$PLANNER/$SPLIT/$BRANCH_NAME/$(echo $CKPT_NAME | tr '/: ' '___')_$(TZ='Asia/Shanghai' date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=sequential \
    enable_simulation_progress_bar=true \
    planner.diffusion_planner.config.ego_progress_rerank_topk=1 \
    planner.diffusion_planner.config.ego_progress_rerank_beta=0.0 \
    planner.diffusion_planner.config.token_decode_mode=$TOKEN_DECODE_MODE \
    planner.diffusion_planner.config.token_decode_temperature=$TOKEN_DECODE_TEMPERATURE \
    planner.diffusion_planner.config.token_debug_log_path=$CKPT_DIR/ego_token_debug_${SPLIT}_${TOKEN_DECODE_MODE}_tau${TOKEN_DECODE_TEMPERATURE}.csv \
    hydra.searchpath="[file://$SCRIPT_DIR/diffusion_planner/config/scenario_filter, file://$SCRIPT_DIR/diffusion_planner/config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"

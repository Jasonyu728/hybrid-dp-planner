export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=0,1
export HYDRA_FULL_ERROR=1

# Add PyTorch bundled CUDA libs to LD_LIBRARY_PATH (fixes missing libnvrtc.so)
_torch_lib=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
if [ -n "$_torch_lib" ]; then
    export LD_LIBRARY_PATH="$_torch_lib:${LD_LIBRARY_PATH}"
fi

###################################
# User Configuration Section
###################################
# Set environment variables
export NUPLAN_DEVKIT_ROOT="/root/autodl-tmp/nuplan-devkit"   # e.g. "/data3/yuzhuoyi/nuplan-devkit"
export NUPLAN_DATA_ROOT="/root/autodl-tmp/nuplan-devkit/nuplan/dataset"               # e.g. "/data3/yuzhuoyi/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/root/autodl-tmp/nuplan-devkit/nuplan/dataset/maps"               # e.g. "/data3/yuzhuoyi/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="/root/autodl-tmp/nuplan-devkit/nuplan/dataset/exp"                 # e.g. "/data3/yuzhuoyi/nuplan/exp"

# Dataset split to use
# Options:
#   - "test14-random"
#   - "test14-hard"
#   - "val14"
SPLIT="val14"

# Challenge type
# Options:
#   - "closed_loop_nonreactive_agents"   (NR: 周围车辆不对 ego 做出反应)
#   - "closed_loop_reactive_agents"      (R:  周围车辆会对 ego 做出反应)
CHALLENGE="closed_loop_nonreactive_agents"

# Token / trajectory inference mode.
# 设置为 auto 时，不覆盖 checkpoint 的 args.json。
# 当前默认配套 hybrid continuous 训练的 checkpoint：sim 走 continuous_head 输出。
# 如要复评老的纯 token checkpoint，把下面三个 *_HEAD / OUTPUT_MODE 改回 auto 即可。
USE_TOKEN_CLASSIFIER=auto
TOKEN_SELECTION_MODE=auto
TOKEN_DECODE_MODE=auto
USE_CONTINUOUS_HEAD=true                   # ★ hybrid: 启用连续头
TRAJECTORY_OUTPUT_MODE=continuous_head     # ★ hybrid: sim 输出走连续头（绕过 token 量化）
EGO_PROGRESS_RERANK_TOPK=auto
EGO_PROGRESS_RERANK_BETA=auto
TOKEN_DEBUG_LOG_PATH=""
TRAJECTORY_DEBUG_LOG_PATH=""

###################################


BRANCH_NAME=diffusion_planner_release
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$SCRIPT_DIR:$NUPLAN_DEVKIT_ROOT:$PYTHONPATH"


###################################
# Checkpoint Configuration
# 每次评估前修改 CKPT_DIR 指向对应的训练结果目录。
# 当前是 hybrid 训练的 ckpt 目录占位，开训后会生成具体子目录（带时间戳），需手动填上。
CKPT_NAME="training_hybrid_continuous_token_cls_ce03/<填训练时间戳>"
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
echo "Args file: $ARGS_FILE"
echo "Token mode override: classifier=$USE_TOKEN_CLASSIFIER selection=$TOKEN_SELECTION_MODE decode=$TOKEN_DECODE_MODE"
echo "Trajectory mode override: continuous_head=$USE_CONTINUOUS_HEAD output=$TRAJECTORY_OUTPUT_MODE"
echo "Progress rerank: topk=$EGO_PROGRESS_RERANK_TOPK beta=$EGO_PROGRESS_RERANK_BETA"

if [ ! -f "$ARGS_FILE" ]; then
    echo "ERROR: args.json not found: $ARGS_FILE"
    exit 1
fi

if [ ! -f "$CKPT_FILE" ]; then
    echo "ERROR: checkpoint not found: $CKPT_FILE"
    exit 1
fi

if [ -z "$TRAJECTORY_DEBUG_LOG_PATH" ]; then
    TRAJECTORY_DEBUG_LOG_PATH="$CKPT_DIR/trajectory_debug_${SPLIT}.csv"
fi
if [ -z "$TOKEN_DEBUG_LOG_PATH" ]; then
    TOKEN_DEBUG_LOG_PATH="$CKPT_DIR/ego_token_debug_${SPLIT}.csv"
fi
echo "Token debug log: $TOKEN_DEBUG_LOG_PATH"
echo "Trajectory debug log: $TRAJECTORY_DEBUG_LOG_PATH"

FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=diffusion_planner

CONFIG_OVERRIDES=()
add_config_override() {
    local key="$1"
    local value="$2"
    if [ "$value" != "auto" ]; then
        CONFIG_OVERRIDES+=("++planner.diffusion_planner.config.${key}=${value}")
    fi
}

add_config_override use_token_classifier "$USE_TOKEN_CLASSIFIER"
add_config_override token_selection_mode "$TOKEN_SELECTION_MODE"
add_config_override token_decode_mode "$TOKEN_DECODE_MODE"
add_config_override use_continuous_head "$USE_CONTINUOUS_HEAD"
add_config_override trajectory_output_mode "$TRAJECTORY_OUTPUT_MODE"
add_config_override ego_progress_rerank_topk "$EGO_PROGRESS_RERANK_TOPK"
add_config_override ego_progress_rerank_beta "$EGO_PROGRESS_RERANK_BETA"
add_config_override token_debug_log_path "$TOKEN_DEBUG_LOG_PATH"
add_config_override trajectory_debug_log_path "$TRAJECTORY_DEBUG_LOG_PATH"

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.diffusion_planner.config.args_file=$ARGS_FILE \
    planner.diffusion_planner.ckpt_path=$CKPT_FILE \
    scenario_builder=$SCENARIO_BUILDER \
    scenario_filter=$SPLIT \
    experiment_uid=$PLANNER/$SPLIT/$BRANCH_NAME/$(echo $CKPT_NAME | tr '/: ' '___')_$(TZ='Asia/Shanghai' date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=48 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.2 \
    enable_simulation_progress_bar=true \
    "${CONFIG_OVERRIDES[@]}" \
    hydra.searchpath="[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments  ]"

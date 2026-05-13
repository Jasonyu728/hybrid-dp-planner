#!/usr/bin/env bash
set -e

###################################
# 直接预测 token 编号的训练脚本
# 运行方式：在 Diffusion-Planner 根目录执行
# bash method_token_direct/torch_run.sh
###################################

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2

# 限制每个进程内部的 BLAS/OpenMP 线程数，避免 DataLoader 多进程和 BLAS 线程互相抢 CPU。
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

###################################
# 数据和词表路径
###################################

TRAIN_SET_PATH="/root/autodl-tmp/Diffusion-Planner/npzdata_50000"
TRAIN_SET_LIST_PATH="/root/autodl-tmp/Diffusion-Planner/diffusion_planner_training.json"

# 词表路径必须和 tokenize_npz.py 写入 npz 时使用的词表一致。
EGO_VOCAB_PATH="/root/autodl-tmp/Diffusion-Planner/npz2token_vocab_dataset/ego_vocab_1024.npz"
NBR_VOCAB_PATH="/root/autodl-tmp/Diffusion-Planner/npz2token_vocab_dataset/nbr_vocab_1024.npz"

SAVE_DIR="/root/autodl-tmp/Diffusion-Planner/yzy_output"
EXP_NAME="training_token_direct_cls"

NPROC_PER_NODE=3
MASTER_PORT=22323

###################################
# 训练参数
###################################

TRAIN_ARGS=(
    # 数据集和词表
    --train_set "$TRAIN_SET_PATH"
    --train_set_list "$TRAIN_SET_LIST_PATH"
    --vocab_path "$EGO_VOCAB_PATH"
    --nbr_vocab_path "$NBR_VOCAB_PATH"

    # 日志和 checkpoint
    --save_dir "$SAVE_DIR"
    --name "$EXP_NAME"
    --save_utd 40

    # 数据增强：只用在线扰动，不用离线 token augment。
    --use_data_augment True

    # batch_size 是所有 GPU 合计的总 batch size。
    --batch_size 576
    --num_workers 16
    --train_epochs 1000

    # 本方法只保留两类损失：
    # 1. diffusion loss：让模型输出的连续向量保持稳定
    # 2. token 分类 loss：直接监督模型选对 token 编号
    --alpha_planning_loss 1.0
    --lambda_token_cls_ce 1.0

    # 固定 token embedding，不再让 embedding 跟着模型一起漂移。
    --learnable_token_emb False

    # 开启分类头；sim 阶段也用分类头选 token。
    --use_token_classifier True
    --token_selection_mode classifier

    # 分布式训练
    --ddp True
)

torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="$MASTER_PORT" \
    method_token_direct/train_predictor.py \
    "${TRAIN_ARGS[@]}"

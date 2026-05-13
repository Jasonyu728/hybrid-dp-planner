export CUDA_VISIBLE_DEVICES=0,1,2

# 限制每个 OpenMP/BLAS worker 只使用 1 个线程。
# DataLoader 本身会启动多个进程；如果这里设置过大，容易造成 CPU 过度抢占，
# 也可能触发 libgomp / OpenBLAS 线程数相关问题。
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

###################################
# 用户配置区
###################################

# 默认你已经手动激活了正确的 conda 环境。
# RUN_PYTHON_PATH 仅保留作路径记录；下面实际使用的是 PATH 中的 torchrun。
RUN_PYTHON_PATH="/root/miniconda3/envs/diffusion_planner/bin/python"

# data_process.py 生成的训练 npz 数据路径。
TRAIN_SET_PATH="/root/autodl-tmp/Diffusion-Planner/nuplan_diffusionplanner_large"
TRAIN_SET_LIST_PATH="/root/autodl-tmp/Diffusion-Planner/diffusion_planner_training.json"

# 词表路径必须和 tokenize_npz.py 处理数据时使用的词表完全一致。
# 当前训练用的是 0511 rolling-reference vocab：
#   ego: vocab_size=1024, seg_dim=15, angle_weight=17.1956, md5=73b58c3ae369
#   nbr: vocab_size=1024, seg_dim=15, angle_weight=11.9898, md5=c3b3497ec549
EGO_VOCAB_PATH="/root/autodl-tmp/Diffusion-Planner/npz2token_vocab_dataset/ego_vocab_0511_1024.npz"
NBR_VOCAB_PATH="/root/autodl-tmp/Diffusion-Planner/npz2token_vocab_dataset/nbr_vocab_0511_1024.npz"

# 输出目录和实验名称。
SAVE_DIR="/root/autodl-tmp/Diffusion-Planner/yzy_output"
EXP_NAME="training_hybrid_continuous_token_cls_ce03"

# DDP 配置。NPROC_PER_NODE 应该等于可见 GPU 数量。
NPROC_PER_NODE=3
MASTER_PORT=22323

###################################
# 训练参数
###################################

TRAIN_ARGS=(
    # 数据集和词表。
    --train_set "$TRAIN_SET_PATH"
    --train_set_list "$TRAIN_SET_LIST_PATH"
    --vocab_path "$EGO_VOCAB_PATH"
    --nbr_vocab_path "$NBR_VOCAB_PATH"

    # 日志和 checkpoint 保存。
    --save_dir "$SAVE_DIR"
    --name "$EXP_NAME"
    --save_utd 40

    # 数据增强。
    # use_data_augment=True 表示启用在线状态扰动；train_epoch.py 会对扰动后的轨迹重新 tokenization。
    --use_data_augment True

    # 训练吞吐相关参数。
    # batch_size 是所有 DDP 进程合计的全局 batch size。
    # num_workers 是每个进程的 DataLoader worker 数，总 worker 数 = NPROC_PER_NODE * num_workers。
    --batch_size 576
    --num_workers 16
    --train_epochs 1000

    # ===== Hybrid（连续头 + Token 辅助监督）训练配置 =====
    # 架构：DiT → x0_hat latent ─┬→ continuous_head → 80 帧连续轨迹  (主输出，sim 用)
    #                              └→ classifier     → token IDs       (辅助监督)
    # 目标分数：65~85（5 万数据），85~95（全量数据）

    # 主 loss：扩散 MSE（embedding 空间）。
    # 改回 1.0：之前 3.0 是为了在纯 token 模式下放大 ego 信号；hybrid 模式下连续头
    # 才是主输出，token MSE 重要性下降。
    --alpha_planning_loss 1.0

    # ★ 连续头主损失权重：把它当成"原 DP 的回归损失"。1.0 是基线推荐。
    --lambda_continuous_traj 1.0

    # Token 监督：作为 latent 结构化的 auxiliary。0.3 是温和但有效的权重。
    # 太高（>=1.0）会让 token 反客为主，破坏连续头精度；太低（<=0.1）则 token 监督无效。
    --lambda_token_cls_ce 0.3

    # ★ 启用连续头 + sim 走连续头输出。
    --use_continuous_head True
    --trajectory_output_mode continuous_head

    # Token embedding 设置。
    # learnable_token_emb=False：embedding 完全冻结在 centroid 投影初始化，
    #   稳定、可复现，是 hybrid 模式下的安全起点。
    # learnable_token_emb=True：需要配合 --lambda_emb_commit > 0（推荐 0.25）
    #   才会真正训练 emb；否则虽然 requires_grad=True 但没有 loss 路径更新它。
    --learnable_token_emb False
    --lambda_emb_commit 0.0
    --use_token_classifier True
    --token_selection_mode classifier

    # 分布式训练。
    --ddp True
)

torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="$MASTER_PORT" \
    train_predictor.py \
    "${TRAIN_ARGS[@]}"

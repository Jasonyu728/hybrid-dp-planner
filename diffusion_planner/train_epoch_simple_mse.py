"""
train_epoch_simple_mse.py
=========================
备份版本：方案一（位移加权 Loss）修改前的 train_epoch.py。

Loss 部分使用最简单的 MSE：
    loss_ego = mean((pred_ego - x0_ego)^2)
    loss_nbr = mean((pred_nbr - x0_nbr)^2)
    loss = alpha * loss_ego + loss_nbr

其余逻辑（token 化、SDE 加噪、EMA、梯度裁剪）与方案一修改后版本完全一致。
"""

import torch
import torch.nn as nn


def _tokenize_v5_batch(
    future_traj: torch.Tensor,   # (N, T, 3)  [x, y, heading]
    centroids: torch.Tensor,     # (V, 15)  raw centroids (unscaled)
    angle_weight: float = 3.0,
    k_tokens: int = 16,
    token_step: int = 5,
    n_special: int = 3,
) -> torch.Tensor:               # (N, 18) int64
    """
    PyTorch 版 v5 在线 tokenization，供数据增强后重新对齐 token 目标使用。

    前 token_step 帧 xy 全零的轨迹视为缺失 agent，直接返回全 PAD（id=0）序列。
    angle_weight 需与构建词表时一致（默认 3.0）。
    """
    N = future_traj.shape[0]
    device = future_traj.device

    # 缺失 agent：前 token_step 帧 xy 均为 0
    traj_valid = (future_traj[:, :token_step, :2].abs().sum(dim=(1, 2)) > 1e-6)  # (N,)

    # 预缩放 centroids（角度维度 × angle_weight，与词表编码保持一致）
    cs = centroids.float().clone()
    cs[:, 2::3] *= angle_weight  # (V, 15)

    ids = torch.zeros(N, k_tokens + 2, dtype=torch.long, device=device)   # 默认全 PAD
    ids[traj_valid, 0]  = 1  # BOS
    ids[traj_valid, -1] = 2  # EOS

    x_ref = torch.zeros(N, device=device)
    y_ref = torch.zeros(N, device=device)
    h_ref = torch.zeros(N, device=device)

    for i in range(k_tokens):
        cos_h = torch.cos(h_ref)  # (N,)
        sin_h = torch.sin(h_ref)

        # 构建第 i 个 token 的 15 维查询向量
        sub_list = []
        for j in range(token_step):
            frame_idx = i * token_step + j
            xf = future_traj[:, frame_idx, 0]
            yf = future_traj[:, frame_idx, 1]
            hf = future_traj[:, frame_idx, 2]
            dx_l = cos_h * (xf - x_ref) + sin_h * (yf - y_ref)
            dy_l = -sin_h * (xf - x_ref) + cos_h * (yf - y_ref)
            dh   = torch.atan2(torch.sin(hf - h_ref), torch.cos(hf - h_ref))
            sub_list += [dx_l, dy_l, dh]
        seg = torch.stack(sub_list, dim=1).float()  # (N, 15)

        # 角度缩放后最近邻查找
        seg_sc = seg.clone()
        seg_sc[:, 2::3] *= angle_weight
        dists  = torch.cdist(seg_sc, cs)             # (N, V)
        chosen = dists.argmin(dim=1)                 # (N,)
        ids[traj_valid, i + 1] = (chosen + n_special)[traj_valid]

        # 用 chosen centroid 最后一帧更新 rolling 参考点（与推理 decode 一致）
        dx_c = centroids[chosen, 12].float()
        dy_c = centroids[chosen, 13].float()
        dh_c = centroids[chosen, 14].float()
        x_ref = x_ref + cos_h * dx_c - sin_h * dy_c
        y_ref = y_ref + sin_h * dx_c + cos_h * dy_c
        h_ref = torch.atan2(torch.sin(h_ref + dh_c), torch.cos(h_ref + dh_c))

    return ids


def train_epoch(train_loader, model, optimizer, args, model_ema=None, aug=None):
    """
    Parameters
    ----------
    train_loader : DataLoader
    model        : Diffusion_Planner（或 DDP 包装后的版本）
    optimizer    : AdamW
    args         : 训练参数（含 vocab_path、state_normalizer 等）
    model_ema    : ModelEma（可选）
    aug          : StatePerturbation（可选，数据增强）

    Returns
    -------
    loss_dict    : dict  各项 loss 的 epoch 平均值
    total_loss   : float  总 loss 的 epoch 平均值
    """

    model.train()

    # 获取底层模型（兼容 DDP）
    raw_model     = model.module if hasattr(model, 'module') else model
    sde           = raw_model.sde
    # token embedding 表（固定，不训练）
    ego_token_emb = raw_model.decoder.decoder.ego_token_emb
    nbr_token_emb = raw_model.decoder.decoder.nbr_token_emb
    D             = ego_token_emb.embedding_dim

    # 训练统计
    sum_loss_ego  = 0.0
    sum_loss_nbr  = 0.0
    sum_loss_total = 0.0
    n_batches = 0

    for batch in train_loader:

        # ── 1. 移动到 device ────────────────────────────────────────────
        batch = {
            k: v.to(args.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        B = batch['ego_current_state'].shape[0]
        P = 1 + args.predicted_neighbor_num   # ego + N neighbors

        # ── 2. 数据增强（状态扰动，可选）───────────────────────────────
        if aug is not None:
            batch, aug_ego_future, aug_nbr_future = aug(
                batch,
                batch['ego_agent_future'],
                batch['neighbor_agents_future'],
            )
            # 增强后世界坐标系已重置到扰动 ego 原点；
            # 离线 token IDs 仍是旧坐标系，需在线重新 tokenize 以保持一致。
            with torch.no_grad():
                ego_c = raw_model.decoder.decoder.ego_traj_decoder.centroids  # (V_ego, 15)
                nbr_c = raw_model.decoder.decoder.nbr_traj_decoder.centroids  # (V_nbr, 15)

                batch['ego_token_ids'] = _tokenize_v5_batch(
                    aug_ego_future, ego_c)                                     # (B, 18)

                N_pred = args.predicted_neighbor_num
                nbr_fut = aug_nbr_future[:, :N_pred]                          # (B, N_pred, 80, 3)
                new_nbr_ids = _tokenize_v5_batch(
                    nbr_fut.reshape(B * N_pred, nbr_fut.shape[2], 3), nbr_c,
                ).reshape(B, N_pred, 18)                                       # (B, N_pred, 18)
                batch['neighbor_token_ids'] = batch['neighbor_token_ids'].clone()
                batch['neighbor_token_ids'][:, :N_pred] = new_nbr_ids

        # ── 3. Token IDs → token embedding 向量序列（用于 SDE 加噪）────
        with torch.no_grad():
            ego_motion_ids = batch['ego_token_ids'][:, 1:17]           # (B, 16)
            ego_x0 = ego_token_emb(ego_motion_ids)                     # (B, 16, D)

            nbr_motion_ids = batch['neighbor_token_ids'][
                :, :args.predicted_neighbor_num, 1:17
            ]                                                           # (B, P-1, 16)
            B_n, P1, K = nbr_motion_ids.shape
            nbr_x0 = nbr_token_emb(
                nbr_motion_ids.reshape(B_n * P1, K)
            ).reshape(B_n, P1, K, D)                                   # (B, P-1, 16, D)

        # ── 4. 构建干净 embedding x0: (B, P, 16*D) ──────────────────────
        all_x0 = torch.cat([
            ego_x0.unsqueeze(1),   # (B, 1, 16, D)
            nbr_x0,                # (B, P-1, 16, D)
        ], dim=1)                  # (B, P, 16, D)
        x0 = all_x0.reshape(B, P, -1)   # (B, P, 16*D)

        # ── 5. 扩散加噪 ─────────────────────────────────────────────────
        t = torch.rand(B, device=args.device)

        mean_coeff = sde.marginal_prob(torch.ones(B, device=args.device), t)[0][:, None, None]
        std        = sde.marginal_prob_std(t)[:, None, None]
        noise      = torch.randn_like(x0)
        xt         = mean_coeff * x0 + std * noise                    # (B, P, 16*D)

        # ── 6. 前向传播 ─────────────────────────────────────────────────
        inputs = {
            **batch,
            'sampled_trajectories': xt,
            'diffusion_time':       t,
        }

        _, decoder_outputs = model(inputs)
        pred = decoder_outputs['score'].reshape(B, P, -1)              # (B, P, 16*D)

        # ── 7. MSE Loss（简单版，无加权）────────────────────────────────
        loss_ego = torch.mean((pred[:, 0] - x0[:, 0]) ** 2)
        loss_nbr = torch.mean((pred[:, 1:] - x0[:, 1:]) ** 2)
        loss = args.alpha_planning_loss * loss_ego + loss_nbr

        # ── 8. 反向传播 + 更新 ──────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        # ── 统计 ────────────────────────────────────────────────────────
        sum_loss_ego   += loss_ego.item()
        sum_loss_nbr   += loss_nbr.item()
        sum_loss_total += loss.item()
        n_batches      += 1

    loss_dict = {
        'ego_planning_loss':        sum_loss_ego   / max(n_batches, 1),
        'neighbor_prediction_loss': sum_loss_nbr   / max(n_batches, 1),
        'loss':                     sum_loss_total / max(n_batches, 1),
    }
    total_loss = sum_loss_total / max(n_batches, 1)

    return loss_dict, total_loss

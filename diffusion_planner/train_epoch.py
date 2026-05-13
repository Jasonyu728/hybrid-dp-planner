"""
Training loop for tokenized Diffusion Planner.
"""

from contextlib import nullcontext

import torch
import torch.nn.functional as F


# 判定一帧 xy 是否"非全零"的阈值。原始 npz 里 invalid 帧 xy 精确为 0；
# data_augmentation 后的 ego_future 经 quintic 插值可能引入极小数值噪声，所以
# 用一个 epsilon 而不是严格等于 0。与离线 np.all(traj[:, :2] == 0) 行为等价。
_ZERO_XY_EPS = 1e-6


def _tokenize_v5_batch(
    future_traj: torch.Tensor,
    centroids: torch.Tensor,
    scaled_centroids: torch.Tensor = None,
    angle_weight: float = 3.0,
    k_tokens: int = 16,
    token_step: int = 5,
    n_special: int = 3,
    pad_idx: int = 0,
    bos_idx: int = 1,
    eos_idx: int = 2,
    min_valid_frames: int = None,
) -> torch.Tensor:
    """
    Online v5 tokenization used after state augmentation.

    Parameters
    ----------
    future_traj : (N, T, 3)
    centroids : (V, 15)
    """
    n_batch, t_total = future_traj.shape[:2]
    device = future_traj.device
    max_tokens = min(k_tokens, t_total // token_step)
    if min_valid_frames is None:
        min_valid_frames = token_step

    xy_valid = future_traj[:, :, :2].abs().sum(dim=-1) > _ZERO_XY_EPS
    frame_ids = torch.arange(t_total, device=device).unsqueeze(0).expand(n_batch, -1) + 1
    last_valid = torch.where(xy_valid, frame_ids, torch.zeros_like(frame_ids)).max(dim=1).values
    last_valid = (last_valid // token_step) * token_step
    traj_valid = last_valid >= min_valid_frames
    valid_token_count = (last_valid // token_step).clamp(max=max_tokens)

    centroids = centroids.float()
    if scaled_centroids is None:
        scaled_centroids = centroids.clone()
        scaled_centroids[:, 2::3] *= angle_weight
    else:
        scaled_centroids = scaled_centroids.float()

    ids = torch.full((n_batch, k_tokens + 2), pad_idx, dtype=torch.long, device=device)
    ids[traj_valid, 0] = bos_idx
    ids[traj_valid, -1] = eos_idx

    x_ref = torch.zeros(n_batch, device=device)
    y_ref = torch.zeros(n_batch, device=device)
    h_ref = torch.zeros(n_batch, device=device)

    for i in range(max_tokens):
        block_valid = traj_valid & (valid_token_count > i)
        cos_h = torch.cos(h_ref)
        sin_h = torch.sin(h_ref)

        block = future_traj[:, i * token_step:(i + 1) * token_step]
        dx_g = block[..., 0] - x_ref[:, None]
        dy_g = block[..., 1] - y_ref[:, None]
        dx_l = cos_h[:, None] * dx_g + sin_h[:, None] * dy_g
        dy_l = -sin_h[:, None] * dx_g + cos_h[:, None] * dy_g
        dh = torch.atan2(
            torch.sin(block[..., 2] - h_ref[:, None]),
            torch.cos(block[..., 2] - h_ref[:, None]),
        )
        seg = torch.stack([dx_l, dy_l, dh], dim=-1).reshape(n_batch, token_step * 3).float()
        seg_sc = seg.clone()
        seg_sc[:, 2::3] *= angle_weight
        chosen = torch.cdist(seg_sc, scaled_centroids).argmin(dim=1)
        ids[block_valid, i + 1] = (chosen + n_special)[block_valid]

        # Match tokenize_npz.py::tokenize_future_traj_v5: rolling reference is
        # updated by the selected token centroid, not by the GT last frame.
        dx_c = centroids[chosen, (token_step - 1) * 3 + 0].float()
        dy_c = centroids[chosen, (token_step - 1) * 3 + 1].float()
        dh_c = centroids[chosen, (token_step - 1) * 3 + 2].float()

        x_next = x_ref + cos_h * dx_c - sin_h * dy_c
        y_next = y_ref + sin_h * dx_c + cos_h * dy_c
        h_next = torch.atan2(torch.sin(h_ref + dh_c), torch.cos(h_ref + dh_c))

        x_ref = torch.where(block_valid, x_next, x_ref)
        y_ref = torch.where(block_valid, y_next, y_ref)
        h_ref = torch.where(block_valid, h_next, h_ref)

    return ids


def _token_agent_valid_mask(token_ids: torch.Tensor, bos_idx: int = 1) -> torch.Tensor:
    """Valid tokenized agents start with BOS; absent agents stay all-PAD=0."""
    return token_ids[:, 0] == bos_idx


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    denom = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / denom


def _xyh_to_xycs(future_traj: torch.Tensor) -> torch.Tensor:
    heading = future_traj[..., 2:3]
    return torch.cat(
        [future_traj[..., :2], torch.cos(heading), torch.sin(heading)],
        dim=-1,
    )


def _token_classifier_ce(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    mask: torch.Tensor = None,
    n_special: int = 3,
):
    valid = token_ids >= n_special
    if mask is not None:
        valid = valid & mask

    if valid.sum() == 0:
        zero = logits.sum() * 0.0
        return zero, zero.detach()

    labels = (token_ids - n_special).clamp(min=0, max=logits.shape[-1] - 1)
    valid_flat = valid.reshape(-1)
    logits_flat = logits.reshape(-1, logits.shape[-1])[valid_flat]
    labels_flat = labels.reshape(-1)[valid_flat]
    loss = F.cross_entropy(logits_flat.float(), labels_flat, reduction="mean")

    with torch.no_grad():
        pred_flat = logits_flat.argmax(dim=-1)
        acc = (pred_flat == labels_flat).to(logits_flat.dtype).mean()

    return loss, acc


def _ego_classifier_progress_metrics(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    centroids: torch.Tensor,
    slow_threshold: float = 0.2,
    n_special: int = 3,
):
    valid = token_ids >= n_special
    if valid.sum() == 0:
        zero = logits.sum() * 0.0
        return zero.detach(), zero.detach(), zero.detach()

    progress_col = 12 if centroids.shape[1] >= 15 else 0
    progress_w = centroids[:, progress_col].detach().to(
        device=logits.device,
        dtype=torch.float32,
    )
    pred_ids = logits.argmax(dim=-1).clamp(0, centroids.shape[0] - 1)
    pred_dx = progress_w[pred_ids]
    gt_ids = (token_ids - n_special).clamp(0, centroids.shape[0] - 1)
    gt_dx = progress_w[gt_ids]
    mask = valid.to(pred_dx.dtype)
    denom = mask.sum(dim=1).clamp(min=1.0)
    pred_total_dx = (pred_dx * mask).sum(dim=1).mean()
    gt_total_dx = (gt_dx * mask).sum(dim=1).mean()
    pred_slow_ratio = (((pred_dx < slow_threshold) & valid).to(pred_dx.dtype).sum(dim=1) / denom).mean()
    return pred_total_dx.detach(), gt_total_dx.detach(), pred_slow_ratio.detach()


def _recover_x0_from_prediction(
    pred: torch.Tensor,
    xt: torch.Tensor,
    x0: torch.Tensor,
    mean_coeff: torch.Tensor,
    std: torch.Tensor,
    model_type: str,
):
    """
    Build the diffusion target and recover an x0 estimate for auxiliary losses.
    """
    if model_type == "x_start":
        target = x0.detach()
        x0_hat = pred
    elif model_type == "score":
        var = std.square().clamp(min=1e-6)
        target = -(xt - mean_coeff * x0.detach()) / var
        x0_hat = (xt + var * pred) / mean_coeff.clamp(min=1e-6)
    else:
        raise ValueError(f"Unsupported diffusion model type: {model_type}")

    return target, x0_hat


def train_epoch(train_loader, model, optimizer, args, model_ema=None, aug=None):
    """Run one training epoch."""
    model.train()

    raw_model = model.module if hasattr(model, "module") else model
    decoder = raw_model.decoder.decoder
    sde = raw_model.sde
    model_type = decoder.dit.model_type
    ego_token_emb = decoder.ego_token_emb
    nbr_token_emb = decoder.nbr_token_emb
    ego_token_classifier = getattr(decoder, "ego_token_classifier", None)
    nbr_token_classifier = getattr(decoder, "nbr_token_classifier", None)
    token_dim = ego_token_emb.embedding_dim

    sum_loss_ego = 0.0
    sum_loss_nbr = 0.0
    sum_loss_cont_ego = 0.0
    sum_loss_cont_nbr = 0.0
    sum_loss_token_cls_ego = 0.0
    sum_loss_token_cls_nbr = 0.0
    sum_ego_token_cls_acc = 0.0
    sum_nbr_token_cls_acc = 0.0
    sum_ego_cls_total_dx = 0.0
    sum_ego_gt_total_dx = 0.0
    sum_ego_cls_slow_ratio = 0.0
    sum_loss_emb_commit = 0.0
    sum_loss_total = 0.0
    n_batches = 0

    for batch in train_loader:
        batch = {
            k: v.to(args.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        batch_size = batch["ego_current_state"].shape[0]
        predicted_neighbor_num = args.predicted_neighbor_num
        num_agents = 1 + predicted_neighbor_num

        if aug is not None:
            batch, aug_ego_future, aug_nbr_future = aug(
                batch,
                batch["ego_agent_future"],
                batch["neighbor_agents_future"],
            )

            with torch.no_grad():
                ego_decoder = decoder.ego_traj_decoder
                nbr_decoder = decoder.nbr_traj_decoder

                batch["ego_token_ids"] = _tokenize_v5_batch(
                    aug_ego_future,
                    ego_decoder.centroids,
                    scaled_centroids=ego_decoder.scaled_centroids,
                    angle_weight=ego_decoder.angle_weight,
                    n_special=ego_decoder.N_SPECIAL,
                    pad_idx=ego_decoder.PAD_IDX,
                    bos_idx=ego_decoder.BOS_IDX,
                    eos_idx=ego_decoder.EOS_IDX,
                )

                nbr_fut = aug_nbr_future[:, :predicted_neighbor_num]
                new_nbr_ids = _tokenize_v5_batch(
                    nbr_fut.reshape(batch_size * predicted_neighbor_num, nbr_fut.shape[2], 3),
                    nbr_decoder.centroids,
                    scaled_centroids=nbr_decoder.scaled_centroids,
                    angle_weight=nbr_decoder.angle_weight,
                    n_special=nbr_decoder.N_SPECIAL,
                    pad_idx=nbr_decoder.PAD_IDX,
                    bos_idx=nbr_decoder.BOS_IDX,
                    eos_idx=nbr_decoder.EOS_IDX,
                    min_valid_frames=nbr_decoder.TOKEN_STEP * 2,
                ).reshape(batch_size, predicted_neighbor_num, 18)

                batch["neighbor_token_ids"] = batch["neighbor_token_ids"].clone()
                batch["neighbor_token_ids"][:, :predicted_neighbor_num] = new_nbr_ids

            ego_future_gt = aug_ego_future[..., :3]
            nbr_future_gt = aug_nbr_future[:, :predicted_neighbor_num, ..., :3]
        else:
            ego_future_gt = batch["ego_agent_future"][..., :3]
            nbr_future_gt = batch["neighbor_agents_future"][:, :predicted_neighbor_num, ..., :3]

        # learnable_token_emb 的真正语义在这里实现：
        #   - emb.weight.requires_grad=False（learnable_token_emb=False）→ 包 no_grad，
        #     x0 是常量，embedding 表完全冻结。
        #   - emb.weight.requires_grad=True  （learnable_token_emb=True ）→ 不包 no_grad，
        #     x0 携带到 embedding.weight 的梯度路径。但 diffusion target = x0.detach()
        #     依然只让 DiT 学习，因此 embedding 实际更新需要下面的 commitment loss
        #     （args.lambda_emb_commit > 0）才会生效。
        emb_learnable = ego_token_emb.weight.requires_grad
        emb_ctx = nullcontext() if emb_learnable else torch.no_grad()

        ego_motion_ids = batch["ego_token_ids"][:, 1:17]
        nbr_motion_ids = batch["neighbor_token_ids"][:, :predicted_neighbor_num, 1:17]
        _, p1, k_tokens = nbr_motion_ids.shape

        with emb_ctx:
            ego_x0 = ego_token_emb(ego_motion_ids)
            nbr_x0 = nbr_token_emb(nbr_motion_ids.reshape(batch_size * p1, k_tokens)).reshape(
                batch_size, p1, k_tokens, token_dim
            )

        all_x0 = torch.cat([ego_x0.unsqueeze(1), nbr_x0], dim=1)
        x0 = all_x0.reshape(batch_size, num_agents, -1)

        nbr_valid_mask = _token_agent_valid_mask(
            batch["neighbor_token_ids"][:, :predicted_neighbor_num].reshape(batch_size * p1, -1),
            bos_idx=decoder.nbr_traj_decoder.BOS_IDX,
        ).reshape(batch_size, p1)

        t = torch.rand(batch_size, device=args.device)
        mean_coeff = sde.marginal_prob(torch.ones(batch_size, device=args.device), t)[0][:, None, None]
        std = sde.marginal_prob_std(t)[:, None, None]
        noise = torch.randn_like(x0)
        xt = mean_coeff * x0.detach() + std * noise

        inputs = {
            **batch,
            "sampled_trajectories": xt,
            "diffusion_time": t,
        }

        # 必须与 sim 路径一致地归一化输入。sim 端 planner.py 在 model(inputs) 前会调
        # args.observation_normalizer(inputs)；这里如果不做，模型训练时看到 raw 米制
        # 数值，sim 时看到归一化后值（x 缩放至 (x-10)/20），输入分布完全不同 → sim 必崩。
        # ObservationNormalizer 只处理 normalization.json 里列出的字段，自动跳过
        # sampled_trajectories / diffusion_time 等 latent 字段，不会误伤 diffusion 状态。
        inputs = args.observation_normalizer(inputs)

        _, decoder_outputs = model(inputs)
        pred = decoder_outputs["score"].reshape(batch_size, num_agents, -1)
        target, x0_hat = _recover_x0_from_prediction(
            pred=pred,
            xt=xt,
            x0=x0,
            mean_coeff=mean_coeff,
            std=std,
            model_type=model_type,
        )

        with torch.no_grad():
            ego_centroids = raw_model.decoder.decoder.ego_traj_decoder.centroids
            raw_ids = (ego_motion_ids - 3).clamp(0, ego_centroids.shape[0] - 1)
            tok_c = ego_centroids[raw_ids]
            tok_dist = (tok_c[:, :, 12].square() + tok_c[:, :, 13].square()).sqrt()
            valid_ego_tokens = (ego_motion_ids >= 3).to(tok_dist.dtype)
            traj_dist = (tok_dist * valid_ego_tokens).sum(dim=1) / valid_ego_tokens.sum(dim=1).clamp(min=1.0)
            scene_w = (1.0 + traj_dist / traj_dist.mean().clamp(min=1e-6)).clamp(max=2.0)
            time_w = torch.ones(16, device=args.device)

        pred_ego = pred[:, 0].reshape(batch_size, 16, token_dim)
        x0_ego = x0[:, 0].reshape(batch_size, 16, token_dim)
        tgt_ego = target[:, 0].reshape(batch_size, 16, token_dim)

        token_mse = ((pred_ego - tgt_ego) ** 2).mean(-1)
        loss_ego = (scene_w.unsqueeze(1) * time_w.unsqueeze(0) * token_mse).mean()

        nbr_mse = ((pred[:, 1:] - target[:, 1:]) ** 2).mean(-1)
        loss_nbr = _masked_mean(nbr_mse, nbr_valid_mask)

        ego_centroids = decoder.ego_traj_decoder.centroids
        x0_hat_ego = x0_hat[:, 0].reshape(batch_size, 16, token_dim)
        x0_hat_nbr = x0_hat[:, 1:].reshape(batch_size * p1, k_tokens, token_dim)
        nbr_token_mask = nbr_valid_mask.unsqueeze(-1).expand_as(nbr_motion_ids)

        if ego_token_classifier is not None and nbr_token_classifier is not None:
            ego_cls_logits = ego_token_classifier(x0_hat_ego)
            nbr_cls_logits = nbr_token_classifier(x0_hat_nbr)
            loss_token_cls_ego, ego_token_cls_acc = _token_classifier_ce(
                ego_cls_logits,
                ego_motion_ids,
            )
            loss_token_cls_nbr, nbr_token_cls_acc = _token_classifier_ce(
                nbr_cls_logits,
                nbr_motion_ids.reshape(batch_size * p1, k_tokens),
                mask=nbr_token_mask.reshape(batch_size * p1, k_tokens),
            )
            ego_cls_total_dx, ego_gt_total_dx, ego_cls_slow_ratio = _ego_classifier_progress_metrics(
                ego_cls_logits,
                ego_motion_ids,
                ego_centroids,
                slow_threshold=0.2,
            )
        else:
            if getattr(args, 'lambda_token_cls_ce', 0.0) > 0.0:
                raise RuntimeError("lambda_token_cls_ce > 0 requires use_token_classifier=True")
            loss_token_cls_ego = x0_hat_ego.sum() * 0.0
            loss_token_cls_nbr = x0_hat_nbr.sum() * 0.0
            ego_token_cls_acc = loss_token_cls_ego.detach()
            nbr_token_cls_acc = loss_token_cls_nbr.detach()
            ego_cls_total_dx = loss_token_cls_ego.detach()
            ego_gt_total_dx = loss_token_cls_ego.detach()
            ego_cls_slow_ratio = loss_token_cls_ego.detach()
        loss_token_cls = loss_token_cls_ego + 0.5 * loss_token_cls_nbr

        zero_metric = x0_hat_ego.sum() * 0.0
        lambda_continuous_traj = getattr(args, 'lambda_continuous_traj', 0.0)
        if lambda_continuous_traj > 0.0:
            if getattr(decoder, "continuous_head", None) is None:
                raise RuntimeError(
                    "lambda_continuous_traj > 0 requires --use_continuous_head True"
                )

            pred_traj = decoder.continuous_from_latent(x0_hat)
            ego_gt_xycs = _xyh_to_xycs(ego_future_gt)
            nbr_gt_xycs = _xyh_to_xycs(nbr_future_gt)

            # ── t-weighting: 解决 continuous_head 在高噪声样本上"塌缩到均值"的问题 ──
            # 推理时 dpm_sampler 输出的是干净的 x0；训练时若所有 t 都 supervise 连续头，
            # 高 t（接近纯噪声）样本会迫使头部产出与输入无关的"均值预测"。
            # 用 (1 - t) 作权重让连续头主要在低 t（接近干净）样本上学习。
            t_weight = (1.0 - t).clamp(min=0.0)
            denom = t_weight.sum().clamp(min=1.0)

            cont_ego_per_sample = F.smooth_l1_loss(
                pred_traj[:, 0], ego_gt_xycs, reduction="none"
            ).mean(dim=(-1, -2))                              # (B,)
            loss_cont_ego = (cont_ego_per_sample * t_weight).sum() / denom

            cont_nbr_per_sample = F.smooth_l1_loss(
                pred_traj[:, 1:], nbr_gt_xycs, reduction="none"
            ).mean(dim=(-1, -2))                              # (B, P1)
            cont_nbr_per_sample = (cont_nbr_per_sample * nbr_valid_mask.to(cont_nbr_per_sample.dtype)
                                   ).sum(dim=1) / nbr_valid_mask.sum(dim=1).clamp(min=1.0)
            loss_cont_nbr = (cont_nbr_per_sample * t_weight).sum() / denom

            loss_continuous = loss_cont_ego + 0.5 * loss_cont_nbr
        else:
            loss_cont_ego = zero_metric
            loss_cont_nbr = zero_metric
            loss_continuous = zero_metric

        # ── VQ-VAE 风格 codebook / commitment loss ─────────────────────────
        # 只在 learnable_token_emb=True 且 lambda_emb_commit>0 时生效。
        # MSE(x0, sg(x0_hat)) 把 embedding 拉向 DiT 的预测方向；
        # detach(x0_hat) 确保这条 loss 只更新 emb，不会反过来影响 DiT。
        lambda_emb_commit = getattr(args, 'lambda_emb_commit', 0.0)
        if emb_learnable and lambda_emb_commit > 0.0:
            x0_ego_grad = x0[:, 0]
            x0_nbr_grad = x0[:, 1:]
            loss_emb_commit_ego = F.mse_loss(x0_ego_grad, x0_hat[:, 0].detach())
            loss_emb_commit_nbr_per = F.mse_loss(
                x0_nbr_grad, x0_hat[:, 1:].detach(), reduction="none"
            ).mean(dim=-1)
            loss_emb_commit_nbr = _masked_mean(loss_emb_commit_nbr_per, nbr_valid_mask)
            loss_emb_commit = loss_emb_commit_ego + 0.5 * loss_emb_commit_nbr
        else:
            loss_emb_commit = zero_metric

        loss = (
            args.alpha_planning_loss * loss_ego
            + loss_nbr
            + getattr(args, 'lambda_token_cls_ce', 0.0) * loss_token_cls
            + lambda_continuous_traj * loss_continuous
            + lambda_emb_commit * loss_emb_commit
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite training loss: "
                f"ego={loss_ego.item()}, nbr={loss_nbr.item()}, "
                f"token_cls={loss_token_cls.item()}, "
                f"continuous={loss_continuous.item()}"
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if model_ema is not None:
            model_ema.update(raw_model)

        sum_loss_ego += loss_ego.item()
        sum_loss_nbr += loss_nbr.item()
        sum_loss_cont_ego += loss_cont_ego.item()
        sum_loss_cont_nbr += loss_cont_nbr.item()
        sum_loss_token_cls_ego += loss_token_cls_ego.item()
        sum_loss_token_cls_nbr += loss_token_cls_nbr.item()
        sum_ego_token_cls_acc += ego_token_cls_acc.item()
        sum_nbr_token_cls_acc += nbr_token_cls_acc.item()
        sum_ego_cls_total_dx += ego_cls_total_dx.item()
        sum_ego_gt_total_dx += ego_gt_total_dx.item()
        sum_ego_cls_slow_ratio += ego_cls_slow_ratio.item()
        sum_loss_emb_commit += loss_emb_commit.item()
        sum_loss_total += loss.item()
        n_batches += 1

    loss_dict = {
        "ego_planning_loss": sum_loss_ego / max(n_batches, 1),
        "neighbor_prediction_loss": sum_loss_nbr / max(n_batches, 1),
        "ego_continuous_traj_loss": sum_loss_cont_ego / max(n_batches, 1),
        "neighbor_continuous_traj_loss": sum_loss_cont_nbr / max(n_batches, 1),
        "continuous_traj_loss": (sum_loss_cont_ego + 0.5 * sum_loss_cont_nbr) / max(n_batches, 1),
        "ego_token_cls_ce_loss": sum_loss_token_cls_ego / max(n_batches, 1),
        "neighbor_token_cls_ce_loss": sum_loss_token_cls_nbr / max(n_batches, 1),
        "token_cls_ce_loss": (sum_loss_token_cls_ego + 0.5 * sum_loss_token_cls_nbr) / max(n_batches, 1),
        "ego_token_cls_acc": sum_ego_token_cls_acc / max(n_batches, 1),
        "neighbor_token_cls_acc": sum_nbr_token_cls_acc / max(n_batches, 1),
        "ego_cls_total_dx": sum_ego_cls_total_dx / max(n_batches, 1),
        "ego_gt_total_dx": sum_ego_gt_total_dx / max(n_batches, 1),
        "ego_cls_slow_ratio": sum_ego_cls_slow_ratio / max(n_batches, 1),
        "emb_commit_loss": sum_loss_emb_commit / max(n_batches, 1),
        "loss": sum_loss_total / max(n_batches, 1),
    }
    total_loss = sum_loss_total / max(n_batches, 1)

    return loss_dict, total_loss

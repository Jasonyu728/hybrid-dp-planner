"""
Direct token-classification training loop.

This version keeps the diffusion target as a fixed token embedding, but uses
the token classifier head as the main supervision for selecting the final
token IDs used in simulation.
"""

import torch
import torch.nn.functional as F


def _tokenize_v5_batch(
    future_traj: torch.Tensor,
    centroids: torch.Tensor,
    scaled_centroids: torch.Tensor = None,
    angle_weight: float = 3.0,
    k_tokens: int = 16,
    token_step: int = 5,
    n_special: int = 3,
) -> torch.Tensor:
    """Online v5 tokenization after state augmentation."""
    n_batch = future_traj.shape[0]
    device = future_traj.device

    traj_valid = future_traj[:, :token_step, :2].abs().sum(dim=(1, 2)) > 1e-6
    centroids = centroids.float()
    if scaled_centroids is None:
        scaled_centroids = centroids.clone()
        scaled_centroids[:, 2::3] *= angle_weight
    else:
        scaled_centroids = scaled_centroids.float()

    ids = torch.zeros(n_batch, k_tokens + 2, dtype=torch.long, device=device)
    ids[traj_valid, 0] = 1
    ids[traj_valid, -1] = 2

    x_ref = torch.zeros(n_batch, device=device)
    y_ref = torch.zeros(n_batch, device=device)
    h_ref = torch.zeros(n_batch, device=device)

    for i in range(k_tokens):
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
        ids[traj_valid, i + 1] = (chosen + n_special)[traj_valid]

        dx_c = centroids[chosen, (token_step - 1) * 3 + 0].float()
        dy_c = centroids[chosen, (token_step - 1) * 3 + 1].float()
        dh_c = centroids[chosen, (token_step - 1) * 3 + 2].float()

        x_next = x_ref + cos_h * dx_c - sin_h * dy_c
        y_next = y_ref + sin_h * dx_c + cos_h * dy_c
        h_next = torch.atan2(torch.sin(h_ref + dh_c), torch.cos(h_ref + dh_c))

        x_ref = torch.where(traj_valid, x_next, x_ref)
        y_ref = torch.where(traj_valid, y_next, y_ref)
        h_ref = torch.where(traj_valid, h_next, h_ref)

    return ids


def _token_agent_valid_mask(token_ids: torch.Tensor) -> torch.Tensor:
    """Valid tokenized agents start with BOS=1; absent agents are all PAD=0."""
    return token_ids[:, 0] == 1


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / denom


def _token_classifier_ce(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    mask: torch.Tensor = None,
    n_special: int = 3,
):
    """Cross entropy between classifier logits and GT token IDs."""
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
    """Metrics only: how much forward progress the classifier-selected tokens imply."""
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

    raw_ids = (token_ids - n_special).clamp(0, centroids.shape[0] - 1)
    gt_dx = progress_w[raw_ids]

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
    """Build diffusion target and recover x0_hat for token classification."""
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
    """Run one epoch for the direct token-classification method."""
    model.train()

    raw_model = model.module if hasattr(model, "module") else model
    decoder = raw_model.decoder.decoder
    sde = raw_model.sde
    model_type = decoder.dit.model_type
    ego_token_emb = decoder.ego_token_emb
    nbr_token_emb = decoder.nbr_token_emb
    ego_token_classifier = getattr(decoder, "ego_token_classifier", None)
    nbr_token_classifier = getattr(decoder, "nbr_token_classifier", None)

    if ego_token_classifier is None or nbr_token_classifier is None:
        raise RuntimeError(
            "method_token_direct requires --use_token_classifier True "
            "and --token_selection_mode classifier."
        )

    token_dim = ego_token_emb.embedding_dim

    sum_loss_ego = 0.0
    sum_loss_nbr = 0.0
    sum_loss_token_cls_ego = 0.0
    sum_loss_token_cls_nbr = 0.0
    sum_ego_token_cls_acc = 0.0
    sum_nbr_token_cls_acc = 0.0
    sum_ego_cls_total_dx = 0.0
    sum_ego_gt_total_dx = 0.0
    sum_ego_cls_slow_ratio = 0.0
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
                )
                nbr_fut = aug_nbr_future[:, :predicted_neighbor_num]
                batch["neighbor_token_ids"] = batch["neighbor_token_ids"].clone()
                batch["neighbor_token_ids"][:, :predicted_neighbor_num] = _tokenize_v5_batch(
                    nbr_fut.reshape(batch_size * predicted_neighbor_num, nbr_fut.shape[2], 3),
                    nbr_decoder.centroids,
                    scaled_centroids=nbr_decoder.scaled_centroids,
                    angle_weight=nbr_decoder.angle_weight,
                ).reshape(batch_size, predicted_neighbor_num, 18)

        with torch.no_grad():
            ego_motion_ids = batch["ego_token_ids"][:, 1:17]
            ego_x0 = ego_token_emb(ego_motion_ids)

            nbr_motion_ids = batch["neighbor_token_ids"][:, :predicted_neighbor_num, 1:17]
            _, p1, k_tokens = nbr_motion_ids.shape
            nbr_x0 = nbr_token_emb(nbr_motion_ids.reshape(batch_size * p1, k_tokens)).reshape(
                batch_size, p1, k_tokens, token_dim
            )

        all_x0 = torch.cat([ego_x0.unsqueeze(1), nbr_x0], dim=1)
        x0 = all_x0.reshape(batch_size, num_agents, -1)

        nbr_valid_mask = _token_agent_valid_mask(
            batch["neighbor_token_ids"][:, :predicted_neighbor_num].reshape(batch_size * p1, -1)
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

        pred_ego = pred[:, 0].reshape(batch_size, 16, token_dim)
        tgt_ego = target[:, 0].reshape(batch_size, 16, token_dim)
        loss_ego = ((pred_ego - tgt_ego) ** 2).mean()

        nbr_mse = ((pred[:, 1:] - target[:, 1:]) ** 2).mean(-1)
        loss_nbr = _masked_mean(nbr_mse, nbr_valid_mask)

        x0_hat_ego = x0_hat[:, 0].reshape(batch_size, 16, token_dim)
        x0_hat_nbr = x0_hat[:, 1:].reshape(batch_size * p1, k_tokens, token_dim)

        nbr_token_mask = nbr_valid_mask.unsqueeze(-1).expand_as(nbr_motion_ids)
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
        loss_token_cls = loss_token_cls_ego + 0.5 * loss_token_cls_nbr

        ego_cls_total_dx, ego_gt_total_dx, ego_cls_slow_ratio = _ego_classifier_progress_metrics(
            ego_cls_logits,
            ego_motion_ids,
            decoder.ego_traj_decoder.centroids,
        )

        loss = (
            args.alpha_planning_loss * loss_ego
            + loss_nbr
            + args.lambda_token_cls_ce * loss_token_cls
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite training loss: "
                f"ego={loss_ego.item()}, nbr={loss_nbr.item()}, "
                f"token_cls={loss_token_cls.item()}"
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if model_ema is not None:
            model_ema.update(raw_model)

        sum_loss_ego += loss_ego.item()
        sum_loss_nbr += loss_nbr.item()
        sum_loss_token_cls_ego += loss_token_cls_ego.item()
        sum_loss_token_cls_nbr += loss_token_cls_nbr.item()
        sum_ego_token_cls_acc += ego_token_cls_acc.item()
        sum_nbr_token_cls_acc += nbr_token_cls_acc.item()
        sum_ego_cls_total_dx += ego_cls_total_dx.item()
        sum_ego_gt_total_dx += ego_gt_total_dx.item()
        sum_ego_cls_slow_ratio += ego_cls_slow_ratio.item()
        sum_loss_total += loss.item()
        n_batches += 1

    loss_dict = {
        "ego_planning_loss": sum_loss_ego / max(n_batches, 1),
        "neighbor_prediction_loss": sum_loss_nbr / max(n_batches, 1),
        "ego_token_cls_ce_loss": sum_loss_token_cls_ego / max(n_batches, 1),
        "neighbor_token_cls_ce_loss": sum_loss_token_cls_nbr / max(n_batches, 1),
        "token_cls_ce_loss": (sum_loss_token_cls_ego + 0.5 * sum_loss_token_cls_nbr) / max(n_batches, 1),
        "ego_token_cls_acc": sum_ego_token_cls_acc / max(n_batches, 1),
        "neighbor_token_cls_acc": sum_nbr_token_cls_acc / max(n_batches, 1),
        "ego_cls_total_dx": sum_ego_cls_total_dx / max(n_batches, 1),
        "ego_gt_total_dx": sum_ego_gt_total_dx / max(n_batches, 1),
        "ego_cls_slow_ratio": sum_ego_cls_slow_ratio / max(n_batches, 1),
        "loss": sum_loss_total / max(n_batches, 1),
    }
    total_loss = sum_loss_total / max(n_batches, 1)

    return loss_dict, total_loss

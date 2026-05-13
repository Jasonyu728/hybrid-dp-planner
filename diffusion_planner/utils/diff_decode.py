"""
Differentiable token → trajectory decoder shared between training and guidance.
"""
import torch
import torch.nn.functional as F


def smooth_trajectory_xyh(
    traj_xyh: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 2.0,
) -> torch.Tensor:
    """
    Apply the same Gaussian smoothing used by inference-time token decoding.
    """
    if traj_xyh.numel() == 0:
        return traj_xyh

    n_batch, t_horizon, _ = traj_xyh.shape
    device = traj_xyh.device
    dtype = traj_xyh.dtype

    k = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    kernel = torch.exp(-k ** 2 / (2.0 * sigma ** 2))
    kernel = (kernel / kernel.sum()).view(1, 1, kernel_size)
    pad = kernel_size // 2

    xy = traj_xyh[:, :, :2].permute(0, 2, 1)
    xy = F.pad(xy, (pad, pad), mode="reflect")
    xy = F.conv1d(xy.reshape(n_batch * 2, 1, -1), kernel)
    xy = xy.reshape(n_batch, 2, t_horizon).permute(0, 2, 1)

    heading = traj_xyh[:, :, 2]
    sin_cos = torch.stack([torch.sin(heading), torch.cos(heading)], dim=1)
    sin_cos = F.pad(sin_cos, (pad, pad), mode="reflect")
    sin_cos = F.conv1d(sin_cos.reshape(n_batch * 2, 1, -1), kernel)
    sin_cos = sin_cos.reshape(n_batch, 2, t_horizon)
    heading = torch.atan2(sin_cos[:, 0], sin_cos[:, 1])

    result = traj_xyh.clone()
    result[:, :, :2] = xy
    result[:, :, 2] = heading
    return result


def _differentiable_decode(
    x_emb: torch.Tensor,      # (B, K, D)
    emb_w: torch.Tensor,      # (V, D)   codebook (PAD/BOS/EOS already excluded)
    centroids: torch.Tensor,  # (V, 15)  fixed centroids (5 frames × [dx,dy,dh])
    token_step: int = 5,
    tau: float = 1.0,
) -> torch.Tensor:             # (B, K*token_step, 3)  [x, y, heading], origin at (0,0,0)
    """
    Soft-NN over codebook → soft centroid → rolling decode → global trajectory.

    Fully differentiable w.r.t. x_emb, enabling:
    - Reconstruction loss in train_epoch (prevents codebook collapse)
    - Classifier guidance in token_collision (SAT collision energy)
    """
    B, K, _ = x_emb.shape

    dist     = torch.cdist(x_emb, emb_w)               # (B, K, V)
    soft_w   = torch.softmax(-dist / tau, dim=-1)       # (B, K, V)
    soft_seg = soft_w @ centroids                        # (B, K, 15)
    soft_seg = soft_seg.view(B, K, token_step, 3)        # (B, K, token_step, 3)

    x_ref = torch.zeros(B, device=x_emb.device, dtype=x_emb.dtype)
    y_ref = torch.zeros(B, device=x_emb.device, dtype=x_emb.dtype)
    h_ref = torch.zeros(B, device=x_emb.device, dtype=x_emb.dtype)

    frames = []
    for i in range(K):
        cos_h = torch.cos(h_ref)
        sin_h = torch.sin(h_ref)

        for j in range(token_step):
            dx_l = soft_seg[:, i, j, 0]
            dy_l = soft_seg[:, i, j, 1]
            dh_l = soft_seg[:, i, j, 2]
            x_g = x_ref + cos_h * dx_l - sin_h * dy_l
            y_g = y_ref + sin_h * dx_l + cos_h * dy_l
            h_g = h_ref + dh_l
            frames.append(torch.stack([x_g, y_g, h_g], dim=-1))

        # Rolling reference: use last sub-frame of this token
        dx_last = soft_seg[:, i, -1, 0]
        dy_last = soft_seg[:, i, -1, 1]
        dh_last = soft_seg[:, i, -1, 2]
        x_ref = x_ref + cos_h * dx_last - sin_h * dy_last
        y_ref = y_ref + sin_h * dx_last + cos_h * dy_last
        h_ref = torch.atan2(torch.sin(h_ref + dh_last), torch.cos(h_ref + dh_last))

    return torch.stack(frames, dim=1)   # (B, K*token_step, 3)

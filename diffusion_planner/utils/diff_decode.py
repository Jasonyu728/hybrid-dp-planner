"""
Differentiable token → trajectory decoder shared between training and guidance.
"""
import torch


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

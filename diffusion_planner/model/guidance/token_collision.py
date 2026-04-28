"""
Token-based collision avoidance guidance for DiffusionPlanner.

The DiT denoises token embeddings x ∈ R^[B,P,K*D].
Gradient path: energy → SAT distances → ego_corners → ego_traj4
             → _differentiable_decode → x[:,0]  (ego only; neighbors detached)
"""
import torch

from diffusion_planner.utils.diff_decode import _differentiable_decode
from diffusion_planner.model.guidance.collision import (
    batch_signed_distance_rect,
    center_rect_to_points,
    ego_size,
    COG_TO_REAR,
    CLIP_DISTANCE,
    INFLATION,
)

K_TOKENS = 16


def token_collision_guidance_fn(x, t, cond, inputs, **kwargs):
    """
    x       : (B, P, K*D)  token embeddings with requires_grad=True
    t       : scalar tensor, diffusion time in [0, 1]
    inputs  : batch dict, already inverse-normalized by TokenGuidanceWrapper
    kwargs  : ego_emb_w, nbr_emb_w, ego_centroids, nbr_centroids (from decoder)

    Returns : scalar energy  (negative penalty; less collision → less negative)
    """
    B, P, KD = x.shape
    D     = KD // K_TOKENS
    P_nbr = P - 1

    # Time gate: only guide during late denoising (matches original collision.py)
    t_val = t.item() if (hasattr(t, 'item') and t.numel() == 1) else t.mean().item()
    if not (0.005 < t_val < 0.1):
        return (x * 0).sum()   # zero energy, zero gradient

    ego_emb_w    = kwargs['ego_emb_w']      # (V_ego, D)
    nbr_emb_w    = kwargs['nbr_emb_w']      # (V_nbr, D)
    ego_centroids = kwargs['ego_centroids']  # (V_ego, 15)
    nbr_centroids = kwargs['nbr_centroids']  # (V_nbr, 15)

    # ── Decode ego trajectory (differentiable) ──────────────────────────────
    x_ego    = x[:, 0].reshape(B, K_TOKENS, D)
    traj_ego = _differentiable_decode(x_ego, ego_emb_w, ego_centroids)  # (B, 80, 3)

    ego_h   = traj_ego[:, :, 2]
    ego_cos = torch.cos(ego_h)
    ego_sin = torch.sin(ego_h)
    # Shift from rear-axle reference to COG (matches original collision.py)
    ego_x = traj_ego[:, :, 0] + ego_cos * COG_TO_REAR
    ego_y = traj_ego[:, :, 1] + ego_sin * COG_TO_REAR
    ego_traj4 = torch.stack([ego_x, ego_y, ego_cos, ego_sin], dim=-1)  # (B, T, 4)

    # ── Decode neighbor trajectories (detached — gradient only on ego) ───────
    with torch.no_grad():
        x_nbr    = x[:, 1:].detach().reshape(B * P_nbr, K_TOKENS, D)
        traj_nbr = _differentiable_decode(
            x_nbr, nbr_emb_w, nbr_centroids
        ).reshape(B, P_nbr, 80, 3)                                       # (B, P_nbr, T, 3)
        nbr_h   = traj_nbr[:, :, :, 2]
        nbr_cos = torch.cos(nbr_h)
        nbr_sin = torch.sin(nbr_h)
        nbr_traj4 = torch.stack(
            [traj_nbr[..., 0], traj_nbr[..., 1], nbr_cos, nbr_sin], dim=-1
        )                                                                  # (B, P_nbr, T, 4)

    # ── Sizes ────────────────────────────────────────────────────────────────
    T      = 80
    ego_lw = torch.tensor(ego_size, device=x.device, dtype=x.dtype)      # (2,) [l, w]
    nbr_lw = inputs["neighbor_agents_past"][:, :P_nbr, -1, [7, 6]].to(x.dtype)  # (B, P_nbr, 2)

    # ── Build bounding boxes and compute corners ─────────────────────────────
    ego_bbox6 = torch.cat([
        ego_traj4,
        ego_lw[None, None, :].expand(B, T, -1) + INFLATION,
    ], dim=-1)  # (B, T, 6)

    nbr_bbox6 = torch.cat([
        nbr_traj4,
        nbr_lw[:, :, None, :].expand(-1, -1, T, -1) + INFLATION,
    ], dim=-1)  # (B, P_nbr, T, 6)

    ego_corners = center_rect_to_points(ego_bbox6.reshape(-1, 6)).reshape(B,      T, 4, 2)
    nbr_corners = center_rect_to_points(nbr_bbox6.reshape(-1, 6)).reshape(B, P_nbr, T, 4, 2)

    # ── Select valid (present) neighbors ────────────────────────────────────
    neighbor_current_mask = inputs["neighbor_current_mask"]  # (B, P_nbr) True=absent
    valid   = ~neighbor_current_mask                          # (B, P_nbr) True=present
    valid_T = valid[:, :, None].expand(-1, -1, T)            # (B, P_nbr, T)

    ego_exp = ego_corners[:, None, :, :, :].expand(-1, P_nbr, -1, -1, -1)
    ego_sel = ego_exp[valid_T].reshape(-1, 4, 2)
    nbr_sel = nbr_corners[valid_T].reshape(-1, 4, 2)

    if ego_sel.shape[0] == 0:
        return (x * 0).sum()

    # ── SAT signed distance → collision penalty ──────────────────────────────
    dists   = batch_signed_distance_rect(ego_sel, nbr_sel)   # (N,)  neg = overlap
    penalty = (1.0 - dists / CLIP_DISTANCE).clamp(min=0)     # (N,)  ≥ 0

    # Negative energy: more overlap → more negative → gradient pushes ego away
    energy = -(penalty.sum() / (B + 1e-5))
    return 3.0 * energy

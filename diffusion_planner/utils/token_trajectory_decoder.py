"""
Decode motion-token sequences back into continuous trajectories.
"""

import numpy as np
import torch
import torch.nn as nn

from diffusion_planner.utils.diff_decode import smooth_trajectory_xyh


class TokenTrajectoryDecoder(nn.Module):
    """Decode `[BOS, t1..t16, EOS]` token IDs into 80-step trajectories."""

    PAD_IDX = 0
    BOS_IDX = 1
    EOS_IDX = 2
    N_SPECIAL = 3
    TOKEN_STEP = 5
    K_TOKENS = 16
    T_FUT = 80

    def __init__(self, vocab_path: str):
        super().__init__()
        data = np.load(vocab_path, allow_pickle=False)
        centroids = torch.from_numpy(data["centroids"].astype(np.float32))
        self.register_buffer("centroids", centroids)
        self.vocab_size = int(data["vocab_size"])
        self.angle_weight = float(data["angle_weight"]) if "angle_weight" in data.files else 3.0
        self.seg_dim = int(data["seg_dim"]) if "seg_dim" in data.files else 3
        scaled_centroids = centroids.clone()
        scaled_centroids[:, 2::3] *= self.angle_weight
        self.register_buffer("scaled_centroids", scaled_centroids)

    def decode_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        token_ids : (N, 18)

        Returns
        -------
        trajectory : (N, 80, 4)
            `[x, y, cos_h, sin_h]` in ego-centric coordinates.
        """
        n_batch = token_ids.shape[0]
        device = token_ids.device

        motion_ids = token_ids[:, 1:17]
        is_special = motion_ids < self.N_SPECIAL
        raw_ids = (motion_ids - self.N_SPECIAL).clamp(0, self.vocab_size - 1)

        segs = self.centroids[raw_ids].clone()
        segs[is_special] = 0.0

        if self.seg_dim == 15:
            traj_xyh = self._decode_v5(segs, n_batch, device)
        else:
            keyframes = self._rollout(segs, device)
            traj_xyh = self._interpolate(keyframes, n_batch, device)

        traj_xyh = self._smooth_trajectory(traj_xyh)

        traj = torch.zeros(n_batch, self.T_FUT, 4, device=device, dtype=traj_xyh.dtype)
        traj[:, :, 0] = traj_xyh[:, :, 0]
        traj[:, :, 1] = traj_xyh[:, :, 1]
        traj[:, :, 2] = torch.cos(traj_xyh[:, :, 2])
        traj[:, :, 3] = torch.sin(traj_xyh[:, :, 2])
        return traj

    def decode_ego_and_neighbors(
        self,
        ego_token_ids: torch.Tensor,
        neighbor_token_ids: torch.Tensor,
        predicted_neighbor_num: int,
    ) -> dict:
        """Decode batched ego and neighbor token sequences."""
        n_batch = ego_token_ids.shape[0]
        ego_future = self.decode_tokens(ego_token_ids)

        nbr_tok = neighbor_token_ids[:, :predicted_neighbor_num]
        p1 = nbr_tok.shape[1]
        nbr_future = self.decode_tokens(nbr_tok.reshape(n_batch * p1, 18)).reshape(
            n_batch, p1, self.T_FUT, 4
        )

        return {"ego_future": ego_future, "neighbor_future": nbr_future}

    def _smooth_trajectory(self, traj_xyh: torch.Tensor) -> torch.Tensor:
        return smooth_trajectory_xyh(traj_xyh)

    def _rollout(self, segs: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Decode legacy 3D centroids into 16 keyframes."""
        n_batch = segs.shape[0]
        keyframes = torch.zeros(n_batch, self.K_TOKENS, 3, device=device, dtype=segs.dtype)
        x = torch.zeros(n_batch, device=device, dtype=segs.dtype)
        y = torch.zeros(n_batch, device=device, dtype=segs.dtype)
        h = torch.zeros(n_batch, device=device, dtype=segs.dtype)

        for i in range(self.K_TOKENS):
            dx_l = segs[:, i, 0]
            dy_l = segs[:, i, 1]
            dh = segs[:, i, 2]
            cos_h = torch.cos(h)
            sin_h = torch.sin(h)
            x = x + cos_h * dx_l - sin_h * dy_l
            y = y + sin_h * dx_l + cos_h * dy_l
            h = torch.atan2(torch.sin(h + dh), torch.cos(h + dh))
            keyframes[:, i] = torch.stack([x, y, h], dim=-1)

        return keyframes

    def _interpolate(self, keyframes: torch.Tensor, n_batch: int, device: torch.device) -> torch.Tensor:
        """Linearly upsample 16 keyframes to 80 frames."""
        traj = torch.zeros(n_batch, self.T_FUT, 3, device=device, dtype=keyframes.dtype)
        anchors = torch.cat([torch.zeros(n_batch, 1, 3, device=device, dtype=keyframes.dtype), keyframes], dim=1)

        for i in range(self.K_TOKENS):
            start = anchors[:, i]
            end = anchors[:, i + 1]
            for j in range(self.TOKEN_STEP):
                alpha = (j + 1) / self.TOKEN_STEP
                frame_idx = i * self.TOKEN_STEP + j
                traj[:, frame_idx, :2] = start[:, :2] + alpha * (end[:, :2] - start[:, :2])
                dh = torch.atan2(torch.sin(end[:, 2] - start[:, 2]), torch.cos(end[:, 2] - start[:, 2]))
                traj[:, frame_idx, 2] = start[:, 2] + alpha * dh

        return traj

    def _decode_v5(self, segs: torch.Tensor, n_batch: int, device: torch.device) -> torch.Tensor:
        """Decode 15D centroids that store five local sub-frames per token."""
        traj = torch.zeros(n_batch, self.T_FUT, 3, device=device, dtype=segs.dtype)
        x_ref = torch.zeros(n_batch, device=device, dtype=segs.dtype)
        y_ref = torch.zeros(n_batch, device=device, dtype=segs.dtype)
        h_ref = torch.zeros(n_batch, device=device, dtype=segs.dtype)

        for i in range(self.K_TOKENS):
            cos_h = torch.cos(h_ref)
            sin_h = torch.sin(h_ref)

            for j in range(self.TOKEN_STEP):
                dx_l = segs[:, i, j * 3 + 0]
                dy_l = segs[:, i, j * 3 + 1]
                dh = segs[:, i, j * 3 + 2]

                x_f = x_ref + cos_h * dx_l - sin_h * dy_l
                y_f = y_ref + sin_h * dx_l + cos_h * dy_l
                h_f = torch.atan2(torch.sin(h_ref + dh), torch.cos(h_ref + dh))

                frame_idx = i * self.TOKEN_STEP + j
                traj[:, frame_idx, 0] = x_f
                traj[:, frame_idx, 1] = y_f
                traj[:, frame_idx, 2] = h_f

            dx_last = segs[:, i, (self.TOKEN_STEP - 1) * 3 + 0]
            dy_last = segs[:, i, (self.TOKEN_STEP - 1) * 3 + 1]
            dh_last = segs[:, i, (self.TOKEN_STEP - 1) * 3 + 2]
            x_ref = x_ref + cos_h * dx_last - sin_h * dy_last
            y_ref = y_ref + sin_h * dx_last + cos_h * dy_last
            h_ref = torch.atan2(torch.sin(h_ref + dh_last), torch.cos(h_ref + dh_last))

        return traj

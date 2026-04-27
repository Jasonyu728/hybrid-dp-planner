"""
token_trajectory_decoder.py
============================
将 SMART 风格的 motion token IDs 还原为连续轨迹，
供 DiffusionPlanner 的扩散训练使用。

转换流程
--------
token_ids (B, 18)
  → 取出 16 个 motion token
  → 查 codebook → centroids (B, 16, 3) [dx_local, dy_local, dheading]
  → rollout → 16 个关键帧 (B, 16, 3) [x, y, heading]
  → 插值 → 80 帧完整轨迹 (B, 80, 3)
  → 转格式 → (B, 80, 4) [x, y, cos_h, sin_h]

输出格式与原始 ego_agent_future 格式对齐，可直接替换。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenTrajectoryDecoder(nn.Module):
    """
    将离散 token ID 序列还原为连续轨迹张量。

    Parameters
    ----------
    vocab_path : str
        vocab_512.npz 的路径
    """

    N_SPECIAL  = 3    # PAD=0, BOS=1, EOS=2
    TOKEN_STEP = 5    # 每个 token 覆盖 5 帧（0.5s @ 10Hz）
    K_TOKENS   = 16   # 16 个 motion token（8s 未来）
    T_FUT      = 80   # 80 帧未来轨迹

    def __init__(self, vocab_path: str):
        super().__init__()
        data = np.load(vocab_path, allow_pickle=False)
        centroids = torch.from_numpy(data['centroids'].astype(np.float32))  # (512, 3) or (512, 15)
        self.register_buffer('centroids', centroids)   # 随模型自动移动到 device
        self.vocab_size = int(data['vocab_size'])
        # seg_dim=3：旧版（终点位移）；seg_dim=15：v5（5帧子轨迹）
        self.seg_dim = int(data['seg_dim']) if 'seg_dim' in data.files else 3

    # ── 对外接口 ─────────────────────────────────────────────────────────

    def decode_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        token_ids : (N, 18) int64  [BOS, t1...t16, EOS]

        Returns
        -------
        trajectory : (N, 80, 4) float32  [x, y, cos_h, sin_h]
                     ego-centric 坐标系，与原始 ego_agent_future 格式一致
        """
        N      = token_ids.shape[0]
        device = token_ids.device

        # 取 16 个 motion token（跳过索引 0=BOS 和 17=EOS）
        motion_ids = token_ids[:, 1:17]                               # (N, 16)

        # PAD/特殊 token → 零位移
        is_special = motion_ids < self.N_SPECIAL
        raw_ids    = (motion_ids - self.N_SPECIAL).clamp(0, self.vocab_size - 1)

        # 查 codebook → (N, 16, seg_dim)，seg_dim=3 或 15
        segs = self.centroids[raw_ids].clone()
        segs[is_special] = 0.0

        if self.seg_dim == 15:
            # v5：直接展开 5 帧，无需插值
            traj_xyh = self._decode_v5(segs, N, device)
        else:
            # 旧版：rollout 关键帧 + 线性插值
            keyframes = self._rollout(segs, device)
            traj_xyh  = self._interpolate(keyframes, N, device)

        # 消除 token 边界处的速度跳变（降低 jerk，改善 Comfort）
        traj_xyh = self._smooth_trajectory(traj_xyh)

        # 转换为 [x, y, cos_h, sin_h]
        traj = torch.zeros(N, self.T_FUT, 4, device=device)
        traj[:, :, 0] = traj_xyh[:, :, 0]
        traj[:, :, 1] = traj_xyh[:, :, 1]
        traj[:, :, 2] = torch.cos(traj_xyh[:, :, 2])
        traj[:, :, 3] = torch.sin(traj_xyh[:, :, 2])

        return traj

    def decode_ego_and_neighbors(
        self,
        ego_token_ids: torch.Tensor,       # (B, 18)
        neighbor_token_ids: torch.Tensor,  # (B, N_agents, 18)
        predicted_neighbor_num: int,
    ) -> dict:
        """
        批量解码 ego 和 neighbor token。

        Returns
        -------
        dict:
            'ego_future'      : (B, 80, 4)
            'neighbor_future' : (B, predicted_neighbor_num, 80, 4)
        """
        B  = ego_token_ids.shape[0]
        ego_future = self.decode_tokens(ego_token_ids)               # (B, 80, 4)

        nbr_tok = neighbor_token_ids[:, :predicted_neighbor_num]     # (B, P-1, 18)
        P1      = nbr_tok.shape[1]
        nbr_future = self.decode_tokens(
            nbr_tok.reshape(B * P1, 18)
        ).reshape(B, P1, self.T_FUT, 4)                              # (B, P-1, 80, 4)

        return {'ego_future': ego_future, 'neighbor_future': nbr_future}

    # ── 内部方法 ─────────────────────────────────────────────────────────

    def _smooth_trajectory(self, traj_xyh: torch.Tensor) -> torch.Tensor:
        """
        消除 token 边界（每 5 帧）处的速度不连续，改善 nuPlan Comfort 指标。

        使用 window=5、sigma=1.0 的高斯核：
        - 平滑半径约 ±1 帧（0.1s），仅修正边界跳变
        - 不改变全局轨迹形状，不影响 Drivable/Progress/TTC
        - heading 通过 sin/cos 分量平滑，避免角度跳变

        traj_xyh : (N, 80, 3)  [x, y, heading]
        returns  : (N, 80, 3)  smoothed
        """
        N, T, _ = traj_xyh.shape
        device   = traj_xyh.device
        dtype    = traj_xyh.dtype

        # 高斯核：window=5, sigma=2.0（更强平滑，抑制 token 边界 jerk，改善 Comfort）
        kernel_size = 5
        sigma = 2.0
        k = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
        kernel = torch.exp(-k ** 2 / (2.0 * sigma ** 2))
        kernel = (kernel / kernel.sum()).view(1, 1, kernel_size)
        pad = kernel_size // 2

        # 平滑 x, y 位置（reflect padding 避免端点缩水）
        xy = traj_xyh[:, :, :2].permute(0, 2, 1)           # (N, 2, T)
        xy = F.pad(xy, (pad, pad), mode='reflect')
        xy = F.conv1d(xy.reshape(N * 2, 1, -1), kernel)     # (N*2, 1, T)
        xy = xy.reshape(N, 2, T).permute(0, 2, 1)           # (N, T, 2)

        # 平滑 heading（通过 sin/cos 分量，自动处理角度跳变）
        h  = traj_xyh[:, :, 2]                              # (N, T)
        sc = torch.stack([torch.sin(h), torch.cos(h)], dim=1)  # (N, 2, T)
        sc = F.pad(sc, (pad, pad), mode='reflect')
        sc = F.conv1d(sc.reshape(N * 2, 1, -1), kernel)     # (N*2, 1, T)
        sc = sc.reshape(N, 2, T)
        h  = torch.atan2(sc[:, 0], sc[:, 1])                # (N, T)

        result = traj_xyh.clone()
        result[:, :, 0] = xy[:, :, 0]
        result[:, :, 1] = xy[:, :, 1]
        result[:, :, 2] = h
        return result

    def _rollout(self, segs: torch.Tensor, device) -> torch.Tensor:
        """segs (N,16,3) → keyframes (N,16,3)"""
        N  = segs.shape[0]
        kf = torch.zeros(N, self.K_TOKENS, 3, device=device)
        x  = torch.zeros(N, device=device)
        y  = torch.zeros(N, device=device)
        h  = torch.zeros(N, device=device)

        for i in range(self.K_TOKENS):
            dx_l, dy_l, dh = segs[:, i, 0], segs[:, i, 1], segs[:, i, 2]
            cos_h, sin_h   = torch.cos(h), torch.sin(h)
            x = x + cos_h * dx_l - sin_h * dy_l
            y = y + sin_h * dx_l + cos_h * dy_l
            h = torch.atan2(torch.sin(h + dh), torch.cos(h + dh))
            kf[:, i] = torch.stack([x, y, h], dim=-1)
        return kf

    def _interpolate(self, kf: torch.Tensor, N: int, device) -> torch.Tensor:
        """kf (N,16,3) 关键帧 → traj (N,80,3) 完整轨迹，线性插值"""
        traj    = torch.zeros(N, self.T_FUT, 3, device=device)
        origin  = torch.zeros(N, 1, 3, device=device)
        anchors = torch.cat([origin, kf], dim=1)   # (N, 17, 3)

        for i in range(self.K_TOKENS):
            start = anchors[:, i]
            end   = anchors[:, i + 1]
            for j in range(self.TOKEN_STEP):
                alpha     = (j + 1) / self.TOKEN_STEP
                frame_idx = i * self.TOKEN_STEP + j
                traj[:, frame_idx, :2] = start[:, :2] + alpha * (end[:, :2] - start[:, :2])
                dh = torch.atan2(torch.sin(end[:, 2] - start[:, 2]),
                                 torch.cos(end[:, 2] - start[:, 2]))
                traj[:, frame_idx, 2] = start[:, 2] + alpha * dh
        return traj

    def _decode_v5(self, segs: torch.Tensor, N: int, device) -> torch.Tensor:
        """
        v5 专用：segs (N, 16, 15) → traj (N, 80, 3)

        每个 centroid = [dx0,dy0,dh0, dx1,dy1,dh1, ..., dx4,dy4,dh4]
        dx_j, dy_j, dh_j 均相对于该 token 起点的局部坐标系。

        解码步骤（无插值）：
          对每个 token i：
            1. 将 5 帧的局部位移逐一变换回全局坐标
            2. 直接写入输出轨迹的对应位置
            3. 以最后一帧（j=4）更新 rolling 参考点
        """
        traj  = torch.zeros(N, self.T_FUT, 3, device=device)
        x_ref = torch.zeros(N, device=device)
        y_ref = torch.zeros(N, device=device)
        h_ref = torch.zeros(N, device=device)

        for i in range(self.K_TOKENS):
            cos_h = torch.cos(h_ref)
            sin_h = torch.sin(h_ref)

            for j in range(self.TOKEN_STEP):
                dx_l = segs[:, i, j * 3 + 0]   # 相对 token 起点的局部 x 位移
                dy_l = segs[:, i, j * 3 + 1]   # 相对 token 起点的局部 y 位移
                dh   = segs[:, i, j * 3 + 2]   # 相对 token 起点的朝向差

                # 局部 → 全局坐标变换
                x_f = x_ref + cos_h * dx_l - sin_h * dy_l
                y_f = y_ref + sin_h * dx_l + cos_h * dy_l
                h_f = torch.atan2(torch.sin(h_ref + dh), torch.cos(h_ref + dh))

                frame_idx = i * self.TOKEN_STEP + j
                traj[:, frame_idx, 0] = x_f
                traj[:, frame_idx, 1] = y_f
                traj[:, frame_idx, 2] = h_f

            # 更新 rolling 参考点 = 当前 token 最后一帧（j=4）的全局位置
            dx_last = segs[:, i, (self.TOKEN_STEP - 1) * 3 + 0]
            dy_last = segs[:, i, (self.TOKEN_STEP - 1) * 3 + 1]
            dh_last = segs[:, i, (self.TOKEN_STEP - 1) * 3 + 2]
            x_ref = x_ref + cos_h * dx_last - sin_h * dy_last
            y_ref = y_ref + sin_h * dx_last + cos_h * dy_last
            h_ref = torch.atan2(torch.sin(h_ref + dh_last), torch.cos(h_ref + dh_last))

        return traj

"""
dataset.py
==========
DiffusionPlanner 的 PyTorch Dataset，支持读取 token 字段。

相比原版的修改
--------------
在 __getitem__ 中额外读取：
  ego_token_ids        : (18,)      int64
  neighbor_token_ids   : (32, 18)   int64

这两个字段由 tokenize_npz.py 预先写入 .npz 文件。
其余字段与原版完全一致，不影响模型结构。
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class DiffusionPlannerData(Dataset):
    """
    Parameters
    ----------
    data_path : str | None
        .npz 文件所在目录（与 data_list 二选一或同时使用）
    data_list : str | None
        JSON 文件，内含 .npz 文件名列表（由 data_process.py 生成）
    agent_num : int
        neighbor agent 槽位数（默认 32）
    predicted_neighbor_num : int
        实际参与预测的 neighbor 数（默认 10）
    future_len : int
        未来帧数（默认 80）
    """

    def __init__(
        self,
        data_path: str,
        data_list: str,
        agent_num: int,
        predicted_neighbor_num: int,
        future_len: int,
    ):
        super().__init__()

        self.agent_num              = agent_num
        self.predicted_neighbor_num = predicted_neighbor_num
        self.future_len             = future_len

        # ── 构建文件列表 ────────────────────────────────────────────────
        if data_list is not None and os.path.isfile(data_list):
            with open(data_list, 'r') as f:
                filenames = json.load(f)
            # JSON 中存的是纯文件名，需要拼上目录
            if data_path is not None:
                self.files = [
                    os.path.join(data_path, fn) if not os.path.isabs(fn) else fn
                    for fn in filenames
                ]
            else:
                self.files = filenames
        elif data_path is not None and os.path.isdir(data_path):
            self.files = sorted([
                os.path.join(data_path, fn)
                for fn in os.listdir(data_path)
                if fn.endswith('.npz')
            ])
        else:
            raise ValueError(
                f"无法加载数据：data_path={data_path}, data_list={data_list}"
            )

        # 过滤不存在的文件
        self.files = [f for f in self.files if os.path.isfile(f)]
        print(f"[Dataset] 共加载 {len(self.files)} 个场景")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        data = np.load(self.files[idx], allow_pickle=False)

        item = {
            # ── 当前状态 ────────────────────────────────────────────────
            'ego_current_state': torch.from_numpy(
                data['ego_current_state'].astype(np.float32)),        # (10,)

            # ── 未来轨迹（连续，用于对照；训练时会被 token 版本替换）──
            'ego_agent_future': torch.from_numpy(
                data['ego_agent_future'].astype(np.float32)),         # (80, 3) [x,y,heading]

            # ── 历史轨迹 ────────────────────────────────────────────────
            'neighbor_agents_past': torch.from_numpy(
                data['neighbor_agents_past'].astype(np.float32)),     # (32, 21, 11)
            'neighbor_agents_future': torch.from_numpy(
                data['neighbor_agents_future'].astype(np.float32)),   # (32, 80, 3)

            # ── 静态障碍物 ──────────────────────────────────────────────
            'static_objects': torch.from_numpy(
                data['static_objects'].astype(np.float32)),           # (5, 10)

            # ── 地图 ────────────────────────────────────────────────────
            'lanes': torch.from_numpy(
                data['lanes'].astype(np.float32)),                    # (70, 20, 12)
            'lanes_speed_limit': torch.from_numpy(
                data['lanes_speed_limit'].astype(np.float32)),        # (70, 1)
            'lanes_has_speed_limit': torch.from_numpy(
                data['lanes_has_speed_limit']),                       # (70, 1) bool

            # ── 路线 ────────────────────────────────────────────────────
            'route_lanes': torch.from_numpy(
                data['route_lanes'].astype(np.float32)),              # (25, 20, 12)
            'route_lanes_speed_limit': torch.from_numpy(
                data['route_lanes_speed_limit'].astype(np.float32)),  # (25, 1)
            'route_lanes_has_speed_limit': torch.from_numpy(
                data['route_lanes_has_speed_limit']),                 # (25, 1) bool
        }

        # ── Token 字段（由 tokenize_npz.py 写入）────────────────────────
        if 'ego_token_ids' in data.files:
            item['ego_token_ids'] = torch.from_numpy(
                data['ego_token_ids'].astype(np.int64))               # (18,)
            item['neighbor_token_ids'] = torch.from_numpy(
                data['neighbor_token_ids'].astype(np.int64))          # (32, 18)
        else:
            raise RuntimeError(
                f"{self.files[idx]} 中没有 ego_token_ids 字段。\n"
                "请先运行 tokenize_npz.py 对 cache 目录进行 tokenization。"
            )

        return item

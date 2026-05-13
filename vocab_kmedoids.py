"""
vocab_kmedoids.py
=================
构建 ego / neighbor 专用运动词表，使用 K-Medoids 聚类。

与 vocab.py 的唯一区别
----------------------
  vocab.py          : MiniBatchKMeans，centroid 是簇均值，无内存限制
  vocab_kmedoids.py : K-Medoids，centroid 是真实数据点，需子采样（默认 30k）

其余流程（轨迹切分、过滤、静止占比限制、保存格式）完全一致。

--source 参数控制收集来源：
  ego  → 只收集 ego 轨迹    → 构建 ego 专用词表
  nbr  → 只收集 neighbor     → 构建 neighbor 专用词表
  all  → ego + neighbor 混合（向后兼容）

用法
----
  # 构建 ego 专用词表（512 token）
  python vocab_kmedoids.py \
      --npz_dir /path/to/cache \
      --save    vocab/ego_vocab_512.npz \
      --vocab_size 512 --source ego

  # 构建 neighbor 专用词表（1024 token）
  python vocab_kmedoids.py \
      --npz_dir /path/to/cache \
      --save    vocab/nbr_vocab_1024.npz \
      --vocab_size 1024 --source nbr
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn_extra.cluster import KMedoids

# ── 全局常量（与 vocab.py 保持一致）────────────────────────────────────────
HZ         = 10.0
DT_TOKEN   = 0.5
TOKEN_STEP = int(DT_TOKEN * HZ)    # 5 帧/token
T_FUT      = 80
K_TOKENS   = T_FUT // TOKEN_STEP   # 16 tokens
SEG_DIM    = TOKEN_STEP * 3        # 15 = 5帧 × 3维(dx,dy,dh)

PAD_IDX   = 0
BOS_IDX   = 1
EOS_IDX   = 2
N_SPECIAL = 3

MAX_SPEED_MS  = 40.0
MAX_ANGLE_RAD = np.radians(90.0)


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


# ── 轨迹 → 15维 segments ─────────────────────────────────────────────────────

def extract_segments(future_traj: np.ndarray) -> np.ndarray:
    """(T, 3) → (K, 15)，每个 segment 存储 5 帧在 token 起点坐标系下的局部位移。"""
    T = len(future_traj)
    segs = []
    x_ref, y_ref, h_ref = 0.0, 0.0, 0.0

    for i in range(K_TOKENS):
        end_idx = (i + 1) * TOKEN_STEP - 1
        if end_idx >= T:
            break

        sub = []
        cos_h, sin_h = np.cos(h_ref), np.sin(h_ref)
        for j in range(TOKEN_STEP):
            frame_idx = i * TOKEN_STEP + j
            x_f, y_f, h_f = future_traj[frame_idx]
            dx_g = x_f - x_ref
            dy_g = y_f - y_ref
            dx_l =  cos_h * dx_g + sin_h * dy_g
            dy_l = -sin_h * dx_g + cos_h * dy_g
            dh   = float(_wrap_angle(np.array([h_f - h_ref]))[0])
            sub.extend([dx_l, dy_l, dh])

        segs.append(sub)
        x_ref, y_ref, h_ref = future_traj[end_idx]

    return np.array(segs, dtype=np.float32) if segs else np.zeros((0, SEG_DIM), dtype=np.float32)


def filter_segments(segments: np.ndarray) -> np.ndarray:
    """过滤超速或异常转向的 segment。"""
    DT_FRAME = 1.0 / HZ
    mask = np.ones(len(segments), dtype=bool)

    for j in range(TOKEN_STEP):
        dx_j = segments[:, j * 3 + 0]
        dy_j = segments[:, j * 3 + 1]
        dh_j = segments[:, j * 3 + 2]

        if j == 0:
            step_dx, step_dy, step_dh = dx_j, dy_j, dh_j
        else:
            step_dx = dx_j - segments[:, (j-1)*3+0]
            step_dy = dy_j - segments[:, (j-1)*3+1]
            step_dh = _wrap_angle(dh_j - segments[:, (j-1)*3+2])

        mask &= (np.sqrt(step_dx**2 + step_dy**2) / DT_FRAME <= MAX_SPEED_MS)
        mask &= (np.abs(step_dh) <= MAX_ANGLE_RAD)

    result = segments[mask]
    print(f"[Filter] {len(segments):,} → {len(result):,}  "
          f"（过滤 {len(segments)-len(result):,} 条异常）")
    return result


def cap_stationary(segments: np.ndarray, max_ratio: float = 0.15,
                   seed: int = 42) -> np.ndarray:
    """限制静止 segment 占比，防止词表被静止 token 主导。"""
    total_disp = np.sqrt(segments[:, 12]**2 + segments[:, 13]**2)
    is_stat    = total_disp < (1.0 * DT_TOKEN)

    moving     = segments[~is_stat]
    stationary = segments[is_stat]
    max_stat   = min(int(len(moving) * max_ratio / (1 - max_ratio)), len(stationary))

    rng  = np.random.default_rng(seed)
    kept = stationary[rng.choice(len(stationary), size=max_stat, replace=False)]
    result = np.concatenate([moving, kept], axis=0)
    print(f"[Cap] 运动 {len(moving):,}  静止 {len(stationary):,} → 保留 {max_stat:,}  "
          f"合计 {len(result):,}")
    return result


def collect_segments(npz_dir: str,
                     max_files: Optional[int] = None,
                     max_stationary_ratio: float = 0.15,
                     source: str = 'all') -> np.ndarray:
    """
    遍历 .npz 目录，收集 15D segments 并过滤噪声。

    参数
    ----
    source : 'all' | 'ego' | 'nbr'
        'ego'  — 只收集 ego 轨迹（构建 ego 专用词表）
        'nbr'  — 只收集 neighbor 轨迹（构建 neighbor 专用词表）
        'all'  — ego + neighbor 混合
    """
    assert source in ('all', 'ego', 'nbr'), f"source 须为 all/ego/nbr，实际 {source}"
    files = sorted(Path(npz_dir).rglob("*.npz"))
    if max_files:
        files = files[:max_files]
    print(f"[Vocab] 扫描 {len(files)} 个 .npz 文件（source={source}）...")

    all_segs = []
    for i, fp in enumerate(files):
        if i % 2000 == 0 and i > 0:
            print(f"  进度 {i}/{len(files)}")
        try:
            data = np.load(fp, allow_pickle=False)
        except Exception as e:
            print(f"  跳过 {fp.name}: {e}")
            continue

        if source in ('all', 'ego') and "ego_agent_future" in data:
            s = extract_segments(data["ego_agent_future"])
            if len(s):
                all_segs.append(s)

        if source in ('all', 'nbr') and "neighbor_agents_future" in data:
            nbr = data["neighbor_agents_future"]
            for n in range(nbr.shape[0]):
                if np.allclose(nbr[n, :TOKEN_STEP, :2], 0):
                    continue
                s = extract_segments(nbr[n])
                if len(s):
                    all_segs.append(s)

    if not all_segs:
        raise RuntimeError(f"未从 {npz_dir} 收集到任何 segments（source={source}）")

    raw = np.concatenate(all_segs, axis=0)
    print(f"[Vocab] 收集完毕（{len(raw):,} 条），开始过滤噪声 ...")
    filtered = filter_segments(raw)

    if max_stationary_ratio < 1.0:
        print(f"[Vocab] 限制静止占比 → {max_stationary_ratio*100:.0f}%")
        filtered = cap_stationary(filtered, max_ratio=max_stationary_ratio)

    return filtered


# ── MotionVocabulary（K-Medoids）─────────────────────────────────────────────

class MotionVocabulary:
    """
    15维子轨迹词表，使用 K-Medoids 聚类。

    相比 MiniBatchKMeans（vocab.py）的差异：
      - centroid 是数据集中真实存在的轨迹片段（medoid 性质）
      - 需要构建 N×N 距离矩阵，因此对样本数有限制（默认 max 30k 子采样）
      - 计算更慢
    """
    PAD_IDX   = PAD_IDX
    BOS_IDX   = BOS_IDX
    EOS_IDX   = EOS_IDX
    N_SPECIAL = N_SPECIAL
    SEG_DIM   = SEG_DIM

    def __init__(self, vocab_size: int = 512, angle_weight: float = 3.0, seed: int = 42):
        self.vocab_size   = vocab_size
        self.angle_weight = angle_weight
        self.seed         = seed
        self._centroids: Optional[np.ndarray] = None

    def fit(self, segments: np.ndarray, max_samples: int = 30_000, max_iter: int = 300):
        """
        K-Medoids 聚类。

        参数
        ----
        max_samples : 子采样上限。K-Medoids 需要 N×N 距离矩阵，N=30k 约占 3.6GB。
                      若内存不足可减小此值。
        max_iter    : 最大迭代次数。
        """
        if segments.ndim != 2 or segments.shape[1] != SEG_DIM:
            raise ValueError(f"segments 应为 (N, {SEG_DIM})，实际 {segments.shape}")
        if len(segments) < self.vocab_size:
            raise ValueError(f"segments ({len(segments)}) 少于 vocab_size ({self.vocab_size})")

        # 随机子采样（K-Medoids 内存限制）
        if len(segments) > max_samples:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(segments), size=max_samples, replace=False)
            segments_sub = segments[idx]
            print(f"[Vocab] 随机子采样: {max_samples:,} / {len(segments):,} 条用于 K-Medoids")
        else:
            segments_sub = segments

        X = self._scale(segments_sub)
        print(f"[Vocab] K-Medoids: {len(X):,} × {SEG_DIM}D → {self.vocab_size} 聚类 ...")
        print(f"[Vocab] 距离矩阵约 {len(X)**2 * 4 / 1e9:.1f} GB，请耐心等待 ...")

        km = KMedoids(
            n_clusters=self.vocab_size,
            metric='euclidean',
            method='alternate',
            init='k-medoids++',
            max_iter=max_iter,
            random_state=self.seed,
        )
        km.fit(X)

        # medoid 是真实数据点：用 medoid_indices_ 取出 unscale 前的原始 segment
        self._centroids = segments_sub[km.medoid_indices_].astype(np.float32)
        print(f"[Vocab] 完成。Inertia = {km.inertia_:.4f}")
        return self

    def encode(self, segs: np.ndarray) -> np.ndarray:
        """segs : (N, 15) → token IDs (N,)，含 N_SPECIAL 偏移"""
        self._check()
        X    = self._scale(segs)
        cs   = self._scale(self._centroids)
        diff = X[:, None] - cs[None]
        return np.argmin(np.linalg.norm(diff, axis=-1), axis=-1) + self.N_SPECIAL

    def encode_topk(self, segs: np.ndarray, k: int) -> np.ndarray:
        """segs : (N, 15) → top-k token IDs (N, k)，含 N_SPECIAL 偏移"""
        self._check()
        X    = self._scale(segs)
        cs   = self._scale(self._centroids)
        diff = X[:, None] - cs[None]
        dists = np.linalg.norm(diff, axis=-1)
        return np.argsort(dists, axis=-1)[:, :k] + self.N_SPECIAL

    def save(self, path: str):
        self._check()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path,
                 centroids    = self._centroids,
                 vocab_size   = np.int64(self.vocab_size),
                 angle_weight = np.float32(self.angle_weight),
                 seg_dim      = np.int64(SEG_DIM))
        print(f"[Vocab] 已保存 → {path}  (centroids shape: {self._centroids.shape})")

    @classmethod
    def load(cls, path: str):
        d = np.load(path, allow_pickle=False)
        v = cls(vocab_size=int(d["vocab_size"]), angle_weight=float(d["angle_weight"]))
        v._centroids = d["centroids"].astype(np.float32)
        print(f"[Vocab] 已加载 {v.vocab_size}-token 词表（{SEG_DIM}D，K-Medoids）← {path}")
        return v

    @property
    def centroids(self) -> np.ndarray:
        self._check()
        return self._centroids

    def _scale(self, x: np.ndarray) -> np.ndarray:
        s = x.copy().astype(np.float32)
        s[:, 2::3] *= self.angle_weight
        return s

    def _unscale(self, x: np.ndarray) -> np.ndarray:
        s = x.copy()
        s[:, 2::3] /= self.angle_weight
        return s

    def _check(self):
        if self._centroids is None:
            raise RuntimeError("词表未初始化，请先调用 fit() 或 load()")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="构建运动词表（15D 子轨迹 token，K-Medoids）"
    )
    parser.add_argument("--npz_dir",     required=True, help="npz 数据目录")
    parser.add_argument("--save",        default="./vocab/vocab_512.npz")
    parser.add_argument("--vocab_size",  type=int,   default=512)
    parser.add_argument("--max_files",   type=int,   default=None)
    parser.add_argument("--max_samples", type=int,   default=30_000,
                        help="K-Medoids 子采样上限，默认 30,000（受内存限制）")
    parser.add_argument("--max_iter",    type=int,   default=300)
    parser.add_argument("--angle_weight",         type=float, default=3.0)
    parser.add_argument("--max_stationary_ratio", type=float, default=0.15,
                        help="静止 segment 最大占比，默认 0.15（15%%）")
    parser.add_argument("--source", choices=['all', 'ego', 'nbr'], default='all',
                        help="收集来源：ego=仅ego轨迹, nbr=仅neighbor轨迹, all=全部（默认）")
    args = parser.parse_args()

    segments = collect_segments(
        args.npz_dir, args.max_files,
        max_stationary_ratio=args.max_stationary_ratio,
        source=args.source,
    )
    vocab = MotionVocabulary(vocab_size=args.vocab_size, angle_weight=args.angle_weight)
    vocab.fit(segments, max_samples=args.max_samples, max_iter=args.max_iter)
    vocab.save(args.save)

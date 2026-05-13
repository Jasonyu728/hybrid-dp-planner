"""
vocabulary_v2.py
================
与 v1 相同的 15D 子轨迹 token 设计，但聚类算法换为 K-Medoids。

与 v1 的区别
------------
  v1 : MiniBatchKMeans  → centroid 是簇内均值，不一定是真实数据点
  v2 : K-Medoids        → centroid 保证是数据集中真实存在的轨迹片段

K-Medoids 的内存限制
--------------------
  K-Medoids 需要计算 N×N 的全局距离矩阵：
    30,000 样本 → 30k×30k×4 字节 ≈ 3.6 GB，可行
    1,000,000 样本 → 不可能
  因此必须先随机子采样到 max_samples（默认 30,000）再聚类。
  子采样不影响 centroid 质量，只是从全量数据中随机选代表样本。

用法
----
  python vocabulary_v2.py \
      --npz_dir /path/to/cache \
      --save ./npz2token_dataset/vocab_512_v2.npz \
      --vocab_size 512
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn_extra.cluster import KMedoids

# ── 全局常量 ──────────────────────────────────────────────────────────────
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


# ── 轨迹 → 15维 segments（与 v1 完全相同）────────────────────────────────

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
    """
    过滤噪声 segment。

    对 5 帧中的**每一帧**逐帧位移进行检查，任意一帧超标即过滤整个 segment：
      - 逐帧速度 > 40 m/s（144 km/h）
      - 逐帧转向角 > 90°/0.1s

    说明
    ----
    centroid 存储的 (dx_j, dy_j, dh_j) 均相对于 token 起点，因此：
      - 帧 0 的逐帧位移 = (dx0, dy0)
      - 帧 j 的逐帧位移 = (dx_j - dx_{j-1}, dy_j - dy_{j-1})
    用逐帧位移 / 0.1s 得到每帧的瞬时速度，能准确检出中间帧飞出再飞回的噪声。
    """
    DT_FRAME = 1.0 / HZ   # 0.1s

    n_before = len(segments)
    mask = np.ones(n_before, dtype=bool)

    for j in range(TOKEN_STEP):
        dx_j = segments[:, j * 3 + 0]
        dy_j = segments[:, j * 3 + 1]
        dh_j = segments[:, j * 3 + 2]

        if j == 0:
            step_dx = dx_j
            step_dy = dy_j
            step_dh = dh_j
        else:
            dx_prev = segments[:, (j - 1) * 3 + 0]
            dy_prev = segments[:, (j - 1) * 3 + 1]
            dh_prev = segments[:, (j - 1) * 3 + 2]
            step_dx = dx_j - dx_prev
            step_dy = dy_j - dy_prev
            step_dh = _wrap_angle(dh_j - dh_prev)

        step_speed = np.sqrt(step_dx ** 2 + step_dy ** 2) / DT_FRAME
        mask &= (step_speed <= MAX_SPEED_MS)
        mask &= (np.abs(step_dh) <= MAX_ANGLE_RAD)

    result   = segments[mask]
    n_after  = len(result)
    n_reject = n_before - n_after
    print(f"[Filter] 原始 segments  : {n_before:,}")
    print(f"[Filter] 过滤异常帧     : -{n_reject:,}  (超速或异常转向)")
    print(f"[Filter] 过滤后保留     : {n_after:,}  ({n_after / n_before * 100:.1f}%)")

    spd = np.sqrt(result[:, 12] ** 2 + result[:, 13] ** 2) / DT_TOKEN
    spd_bins   = [0, 1, 3, 6, 10, 20, MAX_SPEED_MS + 1]
    spd_labels = ['静止  <1 m/s ', '低速 1-3 m/s', '慢速 3-6 m/s',
                  '中速 6-10m/s', '快速10-20m/s', '高速20-40m/s']
    print("[Filter] 过滤后速度分布（终点帧平均速度）:")
    for i, label in enumerate(spd_labels):
        cnt = int(((spd >= spd_bins[i]) & (spd < spd_bins[i + 1])).sum())
        pct = cnt / n_after * 100 if n_after else 0
        bar = '█' * min(cnt * 40 // max(n_after, 1), 40)
        print(f"  {label}: {cnt:7,d}  ({pct:5.1f}%)  {bar}")
    return result


def cap_stationary(segments: np.ndarray, max_ratio: float = 0.15, seed: int = 42) -> np.ndarray:
    """
    限制静止 segment 的比例，防止词表被静止 token 主导。

    静止定义：终点帧平均速度 < 1 m/s（总位移 < 0.5m / 0.5s）

    参数
    ----
    max_ratio : 静止 segment 占总数的最大比例，默认 15%
    """
    total_disp = np.sqrt(segments[:, 12] ** 2 + segments[:, 13] ** 2)
    is_stationary = total_disp < (1.0 * DT_TOKEN)   # <1 m/s

    moving     = segments[~is_stationary]
    stationary = segments[is_stationary]

    max_stationary = int(len(moving) * max_ratio / (1 - max_ratio))
    max_stationary = min(max_stationary, len(stationary))

    rng = np.random.default_rng(seed)
    kept_idx = rng.choice(len(stationary), size=max_stationary, replace=False)
    kept_stationary = stationary[kept_idx]

    result = np.concatenate([moving, kept_stationary], axis=0)
    print(f"[Cap] 运动 segments     : {len(moving):,}")
    print(f"[Cap] 静止 segments     : {len(stationary):,}  → 保留 {max_stationary:,} ({max_ratio*100:.0f}%)")
    print(f"[Cap] 合计              : {len(result):,}")
    return result


def collect_segments(npz_dir: str, max_files: Optional[int] = None,
                     max_stationary_ratio: float = 0.15) -> np.ndarray:
    """
    遍历 .npz 目录，收集所有 ego + neighbor 的 15D segments 并过滤噪声。

    参数
    ----
    max_stationary_ratio : 静止 segment 的最大占比，默认 15%
                           设为 1.0 可禁用此限制
    """
    files = sorted(Path(npz_dir).rglob("*.npz"))
    if max_files:
        files = files[:max_files]
    print(f"[Vocab-v2] 扫描 {len(files)} 个 .npz 文件 ...")

    all_segs = []
    for i, fp in enumerate(files):
        if i % 2000 == 0 and i > 0:
            print(f"  进度 {i}/{len(files)}")
        try:
            data = np.load(fp, allow_pickle=False)
        except Exception as e:
            print(f"  跳过 {fp.name}: {e}")
            continue

        if "ego_agent_future" in data:
            s = extract_segments(data["ego_agent_future"])
            if len(s):
                all_segs.append(s)

        if "neighbor_agents_future" in data:
            nbr = data["neighbor_agents_future"]
            for n in range(nbr.shape[0]):
                if np.allclose(nbr[n, :TOKEN_STEP, :2], 0):
                    continue
                s = extract_segments(nbr[n])
                if len(s):
                    all_segs.append(s)

    if not all_segs:
        raise RuntimeError(f"未能从 {npz_dir} 收集到任何 segments")

    raw = np.concatenate(all_segs, axis=0)
    print("[Vocab-v2] 收集完毕，开始过滤噪声 ...")
    filtered = filter_segments(raw)

    if max_stationary_ratio < 1.0:
        print(f"[Vocab-v2] 限制静止 segment 占比 → {max_stationary_ratio*100:.0f}%")
        filtered = cap_stationary(filtered, max_ratio=max_stationary_ratio)

    return filtered


# ── MotionVocabulary（v2：K-Medoids）────────────────────────────────────

class MotionVocabulary:
    """
    15维子轨迹词表，使用 K-Medoids 聚类。

    centroid 形状：(vocab_size, 15)
    每个 centroid 保证是数据集中真实存在的轨迹片段（K-Medoids 的性质）。
    """
    PAD_IDX   = PAD_IDX
    BOS_IDX   = BOS_IDX
    EOS_IDX   = EOS_IDX
    N_SPECIAL = N_SPECIAL
    SEG_DIM   = SEG_DIM    # 15

    def __init__(self, vocab_size: int = 512, angle_weight: float = 3.0, seed: int = 42):
        self.vocab_size   = vocab_size
        self.angle_weight = angle_weight
        self.seed         = seed
        self._centroids: Optional[np.ndarray] = None
        self._predictor: Optional[_Predictor] = None

    # ── 训练 ─────────────────────────────────────────────────────────────

    def fit(self, segments: np.ndarray, max_samples: int = 30_000):
        """
        K-Medoids 聚类。

        流程
        ----
        1. 随机子采样到 max_samples（K-Medoids 需要 N×N 距离矩阵）
        2. 对子采样数据做角度缩放（scale）
        3. K-Medoids 聚类，method='alternate'，init='k-medoids++'
        4. 取 medoid 对应的原始数据点作为 centroid（unscale 后存储）

        K-Medoids vs K-Means
        --------------------
        - K-Means centroid = 簇内均值，可能对应不存在的运动模式
        - K-Medoids centroid = 真实数据点，一定对应真实出现过的轨迹片段
        """
        if segments.ndim != 2 or segments.shape[1] != SEG_DIM:
            raise ValueError(f"segments 应为 (N, {SEG_DIM})，实际 {segments.shape}")
        if len(segments) < self.vocab_size:
            raise ValueError(f"segments ({len(segments)}) 少于 vocab_size ({self.vocab_size})")

        # 随机子采样
        if len(segments) > max_samples:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(segments), size=max_samples, replace=False)
            segments_sub = segments[idx]
            print(f"[Vocab-v2] 随机子采样: {max_samples:,} / {len(segments):,} 条用于 K-Medoids")
        else:
            segments_sub = segments

        X = self._scale(segments_sub)
        print(f"[Vocab-v2] K-Medoids: {len(X):,} × {SEG_DIM}D → {self.vocab_size} 聚类 ...")
        print(f"[Vocab-v2] 使用 alternate 方法，请耐心等待（距离矩阵约 "
              f"{len(X)**2 * 4 / 1e9:.1f} GB）...")

        km = KMedoids(
            n_clusters=self.vocab_size,
            metric='euclidean',
            method='alternate',
            init='k-medoids++',
            max_iter=300,
            random_state=self.seed,
        )
        km.fit(X)

        # medoid_indices 是子采样后 X 的行索引，对应真实数据点
        self._centroids = self._unscale(X[km.medoid_indices_]).astype(np.float32)
        self._predictor = _Predictor(self._centroids, self.angle_weight)
        print(f"[Vocab-v2] 完成。Inertia = {km.inertia_:.4f}")
        print(f"[Vocab-v2] 所有 centroid 均为真实数据点（K-Medoids 保证）")
        return self

    # ── 编码 ─────────────────────────────────────────────────────────────

    def encode(self, segs: np.ndarray) -> np.ndarray:
        """segs : (N, 15) → token IDs (N,)，含 N_SPECIAL 偏移"""
        self._check()
        raw = self._predictor.predict(segs)
        return raw + self.N_SPECIAL

    def encode_topk(self, segs: np.ndarray, k: int) -> np.ndarray:
        """segs : (N, 15) → top-k token IDs (N, k)，含 N_SPECIAL 偏移"""
        self._check()
        X = self._predictor.scale(segs)
        diff  = X[:, None] - self._predictor.cs[None]
        dists = np.linalg.norm(diff, axis=-1)
        topk_raw = np.argsort(dists, axis=-1)[:, :k]
        return topk_raw + self.N_SPECIAL

    # ── 保存 / 加载 ──────────────────────────────────────────────────────

    def save(self, path: str):
        self._check()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path,
                 centroids    = self._centroids,
                 vocab_size   = np.int64(self.vocab_size),
                 angle_weight = np.float32(self.angle_weight),
                 seg_dim      = np.int64(SEG_DIM))
        print(f"[Vocab-v2] 已保存 → {path}  (centroids shape: {self._centroids.shape})")

    @classmethod
    def load(cls, path: str):
        d = np.load(path, allow_pickle=False)
        v = cls(vocab_size=int(d["vocab_size"]), angle_weight=float(d["angle_weight"]))
        v._centroids = d["centroids"].astype(np.float32)
        v._predictor = _Predictor(v._centroids, v.angle_weight)
        print(f"[Vocab-v2] 已加载 {v.vocab_size}-token 词表（{SEG_DIM}D，K-Medoids）← {path}")
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


class _Predictor:
    def __init__(self, centroids: np.ndarray, angle_weight: float):
        self._aw = angle_weight
        self.cs  = self.scale(centroids.copy().astype(np.float32))

    def scale(self, x: np.ndarray) -> np.ndarray:
        s = x.copy().astype(np.float32)
        s[:, 2::3] *= self._aw
        return s

    def predict(self, Xs: np.ndarray) -> np.ndarray:
        Xs_s = self.scale(Xs)
        diff  = Xs_s[:, None] - self.cs[None]
        dists = np.linalg.norm(diff, axis=-1)
        return np.argmin(dists, axis=-1)


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="构建 v2 运动词表（15D 子轨迹 token，K-Medoids）"
    )
    parser.add_argument("--npz_dir",      required=True, help="npz 数据目录")
    parser.add_argument("--save",         default="./npz2token_dataset/vocab_512_v2.npz")
    parser.add_argument("--vocab_size",   type=int,   default=512)
    parser.add_argument("--max_files",    type=int,   default=None)
    parser.add_argument("--max_samples",  type=int,   default=30_000,
                        help="K-Medoids 子采样上限，默认 30000（距离矩阵约 3.6 GB）")
    parser.add_argument("--angle_weight",         type=float, default=3.0)
    parser.add_argument("--max_stationary_ratio", type=float, default=0.15,
                        help="静止 segment 最大占比，默认 0.15（15%%）")
    args = parser.parse_args()

    segments = collect_segments(args.npz_dir, args.max_files,
                                max_stationary_ratio=args.max_stationary_ratio)
    vocab = MotionVocabulary(vocab_size=args.vocab_size, angle_weight=args.angle_weight)
    vocab.fit(segments, max_samples=args.max_samples)
    vocab.save(args.save)

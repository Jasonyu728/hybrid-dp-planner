"""
vocab_v3.py
===========
构建 ego / neighbor 专用运动词表，使用两段式 KMeans 聚类。

相比 vocab.py 的改进
--------------------
  1. 两段式聚类：MiniBatchKMeans warm-up + 全量 KMeans 精炼
     ── 解决近重复 token 问题
  2. 分层采样：按累积位移分桶，平衡速度分布
     ── 解决 nbr 词表低速过密、高速稀疏的双峰问题
  3. 自适应 angle_weight：按通道 std 自动归一化
     ── 解决 ego 词表转向覆盖窄的问题
  4. 收紧帧间转向阈值：90° → 30°
     ── 过滤掉物理上不合理的极端噪声 token
  5. 邻居轨迹截断检测：识别中途消失的 nbr，避免末段补零被当成"减速到停"
  6. 向量化 extract_segments：~20-50x 加速
  7. 分块 encode：避免大 N 时 OOM
  8. 自动诊断报告：聚类完即输出质量自检

用法
----
  # ego 专用词表（auto angle weight）
  python vocab_v3.py \\
      --npz_dir /path/to/cache \\
      --save    vocab/ego_vocab_1024.npz \\
      --vocab_size 1024 --source ego --angle_weight auto

  # nbr 专用词表（开启分层采样平衡分布）
  python vocab_v3.py \\
      --npz_dir /path/to/cache \\
      --save    vocab/nbr_vocab_1024.npz \\
      --vocab_size 1024 --source nbr --angle_weight auto \\
      --stratify --n_buckets 8
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

# ── 全局常量 ─────────────────────────────────────────────────────────────────
HZ         = 10.0
DT_TOKEN   = 0.5
TOKEN_STEP = int(DT_TOKEN * HZ)        # 5 帧/token
T_FUT      = 80
K_TOKENS   = T_FUT // TOKEN_STEP       # 16 tokens
SEG_DIM    = TOKEN_STEP * 3            # 15 = 5帧 × 3维(dx,dy,dh)

PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2
N_SPECIAL = 3

MAX_SPEED_MS  = 40.0
MAX_ANGLE_RAD = np.radians(30.0)       # v2: 90° → v3: 30°（更符合物理约束）


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


# ── 轨迹 → 15维 segments（向量化版本）───────────────────────────────────────

def extract_segments(future_traj: np.ndarray) -> np.ndarray:
    """
    (T, 3) → (K, 15)，每个 segment 存储 5 帧在 token 起点坐标系下的局部位移。
    向量化实现，相比 v2 双重循环约 20-50x 加速。
    """
    T = len(future_traj)
    n_full = T // TOKEN_STEP
    if n_full == 0:
        return np.zeros((0, SEG_DIM), dtype=np.float32)

    traj = future_traj[:n_full * TOKEN_STEP].reshape(n_full, TOKEN_STEP, 3).astype(np.float32)

    # 每段的参考位姿 = 上一段最末帧（第 0 段为原点）
    refs = np.zeros((n_full, 3), dtype=np.float32)
    if n_full > 1:
        refs[1:] = traj[:-1, -1, :]

    cos_h = np.cos(refs[:, 2:3])           # (n_full, 1)
    sin_h = np.sin(refs[:, 2:3])

    dx_g = traj[..., 0] - refs[:, 0:1]     # (n_full, TOKEN_STEP)
    dy_g = traj[..., 1] - refs[:, 1:2]
    dx_l =  cos_h * dx_g + sin_h * dy_g
    dy_l = -sin_h * dx_g + cos_h * dy_g
    dh_l = _wrap_angle(traj[..., 2] - refs[:, 2:3])

    # 交错堆叠为 [dx_0, dy_0, dh_0, dx_1, dy_1, dh_1, ...]
    segs = np.stack([dx_l, dy_l, dh_l], axis=-1).reshape(n_full, SEG_DIM)
    return segs.astype(np.float32)


# ── 数据过滤 ─────────────────────────────────────────────────────────────────

def filter_segments(segments: np.ndarray, verbose: bool = True) -> np.ndarray:
    """过滤超速或异常转向的 segment。"""
    DT_FRAME = 1.0 / HZ
    mask = np.ones(len(segments), dtype=bool)

    speed_killed = 0
    angle_killed = 0

    for j in range(TOKEN_STEP):
        dx_j = segments[:, j * 3 + 0]
        dy_j = segments[:, j * 3 + 1]
        dh_j = segments[:, j * 3 + 2]

        if j == 0:
            step_dx, step_dy, step_dh = dx_j, dy_j, dh_j
        else:
            step_dx = dx_j - segments[:, (j - 1) * 3 + 0]
            step_dy = dy_j - segments[:, (j - 1) * 3 + 1]
            step_dh = _wrap_angle(dh_j - segments[:, (j - 1) * 3 + 2])

        speed_ok = np.sqrt(step_dx**2 + step_dy**2) / DT_FRAME <= MAX_SPEED_MS
        angle_ok = np.abs(step_dh) <= MAX_ANGLE_RAD

        speed_killed += int((mask & ~speed_ok).sum())
        angle_killed += int((mask & ~angle_ok).sum())
        mask &= speed_ok & angle_ok

    result = segments[mask]
    if verbose:
        print(f"[Filter] {len(segments):,} → {len(result):,}  "
              f"（超速 {speed_killed:,} / 超转向 {angle_killed:,}）")
    return result


def cap_stationary(segments: np.ndarray, max_ratio: float = 0.15,
                   seed: int = 42, verbose: bool = True) -> np.ndarray:
    """限制静止 segment 占比，防止词表被静止 token 主导。"""
    total_disp = np.sqrt(segments[:, 12]**2 + segments[:, 13]**2)
    is_stat    = total_disp < (1.0 * DT_TOKEN)

    moving     = segments[~is_stat]
    stationary = segments[is_stat]

    if len(stationary) == 0 or len(moving) == 0:
        return segments

    max_stat = min(int(len(moving) * max_ratio / (1 - max_ratio)), len(stationary))

    rng  = np.random.default_rng(seed)
    kept = stationary[rng.choice(len(stationary), size=max_stat, replace=False)]
    result = np.concatenate([moving, kept], axis=0)
    if verbose:
        print(f"[Cap] 运动 {len(moving):,}  静止 {len(stationary):,} → 保留 {max_stat:,}  "
              f"合计 {len(result):,}")
    return result


def stratified_subsample(segments: np.ndarray, n_buckets: int = 8,
                         target_per_bucket: Optional[int] = None,
                         seed: int = 42, verbose: bool = True) -> np.ndarray:
    """
    按累积位移分桶，每桶等量采样，平衡速度分布。
    解决低速过密导致 KMeans 容量被低速区吞噬的问题。
    """
    if len(segments) == 0:
        return segments

    total_disp = np.sqrt(segments[:, 12]**2 + segments[:, 13]**2)

    edges = np.quantile(total_disp, np.linspace(0, 1, n_buckets + 1))
    edges[-1] += 1e-6
    bucket_ids = np.digitize(total_disp, edges[1:-1])

    counts = np.bincount(bucket_ids, minlength=n_buckets)
    if target_per_bucket is None:
        # 取最小桶大小的 2 倍 与 总量平均 取较小者，避免上采样
        target_per_bucket = int(min(counts.min() * 2, len(segments) // n_buckets))
        target_per_bucket = max(target_per_bucket, counts.min())  # 保底

    rng = np.random.default_rng(seed)
    sampled = []
    for b in range(n_buckets):
        idxs = np.where(bucket_ids == b)[0]
        n_take = min(len(idxs), target_per_bucket)
        if n_take == 0:
            continue
        sampled.append(segments[rng.choice(idxs, n_take, replace=False)])

    result = np.concatenate(sampled, axis=0)
    if verbose:
        bucket_str = " / ".join(f"{c}" for c in counts)
        print(f"[Stratify] 原始桶分布: {bucket_str}")
        print(f"[Stratify] {len(segments):,} → {len(result):,}（{n_buckets} 桶各 ~{target_per_bucket:,}）")
    return result


# ── 邻居轨迹有效性 ──────────────────────────────────────────────────────────

def _is_valid_nbr_traj(traj: np.ndarray, min_valid_frames: int = TOKEN_STEP * 2) -> tuple:
    """
    检测邻居轨迹的有效区段，截掉中途消失（末段全零）的部分。
    返回 (是否可用, 有效末帧索引+1)。
    """
    valid_per_frame = ~np.all(traj[:, :2] == 0, axis=1)
    if valid_per_frame.sum() < min_valid_frames:
        return False, 0

    # 找到最后一个有效帧
    valid_idxs = np.where(valid_per_frame)[0]
    last_valid = int(valid_idxs[-1]) + 1

    # 截到 TOKEN_STEP 整数倍
    last_valid = (last_valid // TOKEN_STEP) * TOKEN_STEP
    if last_valid < min_valid_frames:
        return False, 0
    return True, last_valid


# ── 数据收集 ─────────────────────────────────────────────────────────────────

def collect_segments(npz_dir: str,
                     max_files: Optional[int] = None,
                     max_stationary_ratio: float = 0.15,
                     stratify: bool = False,
                     n_buckets: int = 8,
                     source: str = 'all') -> np.ndarray:
    """
    遍历 .npz 目录，收集 15D segments 并过滤噪声。

    参数
    ----
    source     : 'all' | 'ego' | 'nbr'
    stratify   : 是否对运动 segment 做分层采样（推荐 nbr 词表开启）
    n_buckets  : 分层桶数
    """
    assert source in ('all', 'ego', 'nbr'), f"source 须为 all/ego/nbr，实际 {source}"
    files = sorted(Path(npz_dir).rglob("*.npz"))
    if max_files:
        files = files[:max_files]
    print(f"[Vocab] 扫描 {len(files)} 个 .npz 文件（source={source}）...")

    all_segs = []
    n_nbr_total = 0
    n_nbr_skip  = 0

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
                n_nbr_total += 1
                ok, end = _is_valid_nbr_traj(nbr[n])
                if not ok:
                    n_nbr_skip += 1
                    continue
                s = extract_segments(nbr[n, :end])
                if len(s):
                    all_segs.append(s)

    if not all_segs:
        raise RuntimeError(f"未从 {npz_dir} 收集到任何 segments（source={source}）")

    if source in ('all', 'nbr'):
        print(f"[Vocab] 邻居轨迹有效性: 总 {n_nbr_total:,}, 跳过 {n_nbr_skip:,} "
              f"(无效率 {n_nbr_skip/max(n_nbr_total,1)*100:.1f}%)")

    raw = np.concatenate(all_segs, axis=0)
    print(f"[Vocab] 收集完毕（{len(raw):,} 条），开始过滤噪声 ...")
    filtered = filter_segments(raw)

    if max_stationary_ratio < 1.0:
        print(f"[Vocab] 限制静止占比 → {max_stationary_ratio*100:.0f}%")
        filtered = cap_stationary(filtered, max_ratio=max_stationary_ratio)

    if stratify:
        print(f"[Vocab] 启用分层采样平衡速度分布 ...")
        filtered = stratified_subsample(filtered, n_buckets=n_buckets)

    return filtered


# ── MotionVocabulary（两段式 KMeans）────────────────────────────────────────

class MotionVocabulary:
    PAD_IDX   = PAD_IDX
    BOS_IDX   = BOS_IDX
    EOS_IDX   = EOS_IDX
    N_SPECIAL = N_SPECIAL
    SEG_DIM   = SEG_DIM

    def __init__(self,
                 vocab_size: int = 512,
                 angle_weight: Union[float, str] = 3.0,
                 seed: int = 42):
        self.vocab_size   = vocab_size
        self.angle_weight = angle_weight       # float 或 'auto'
        self.seed         = seed
        self._centroids: Optional[np.ndarray] = None

    # —— 训练 ——————————————————————————————————————————————————————————————

    def fit(self, segments: np.ndarray,
            batch_size: int = 4096,
            max_iter: int = 300,
            n_init: int = 10,
            refine: bool = True,
            refine_max_samples: int = 200_000):
        """
        两段式聚类：
          Stage 1 — MiniBatchKMeans 在全量数据上 warm-up（n_init=10）
          Stage 2 — 全量 KMeans 在子采样上精炼，初始化用 Stage 1 的中心
        """
        if len(segments) < self.vocab_size:
            raise ValueError(f"segments ({len(segments)}) 少于 vocab_size ({self.vocab_size})")

        # 自适应 angle_weight
        if isinstance(self.angle_weight, str) and self.angle_weight.lower() == 'auto':
            raw_std = segments.std(axis=0).reshape(TOKEN_STEP, 3)
            pos_std = raw_std[:, :2].mean()
            ang_std = raw_std[:, 2].mean()
            auto_w = float(pos_std / (ang_std + 1e-6))
            print(f"[Vocab] auto angle_weight = {auto_w:.3f}  "
                  f"(pos_std={pos_std:.3f}, ang_std={ang_std:.3f})")
            self.angle_weight = auto_w

        X = self._scale(segments)

        # Stage 1: MiniBatchKMeans
        print(f"[Vocab] Stage 1: MiniBatchKMeans warm-up "
              f"({len(X):,} × {SEG_DIM}D → {self.vocab_size}, n_init={n_init}) ...")
        km = MiniBatchKMeans(
            n_clusters=self.vocab_size,
            batch_size=batch_size,
            max_iter=max_iter,
            random_state=self.seed,
            n_init=n_init,
            init='k-means++',
            reassignment_ratio=0.01,        # 自动重分配近空簇
            verbose=0,
        )
        km.fit(X)
        print(f"[Vocab] Stage 1 完成。Inertia = {km.inertia_:.4f}")

        if refine:
            # Stage 2: 全量 KMeans 精炼
            rng = np.random.default_rng(self.seed)
            if len(X) > refine_max_samples:
                idx = rng.choice(len(X), refine_max_samples, replace=False)
                X_refine = X[idx]
                print(f"[Vocab] Stage 2: 全量 KMeans 精炼（子采样 {refine_max_samples:,}）...")
            else:
                X_refine = X
                print(f"[Vocab] Stage 2: 全量 KMeans 精炼（{len(X):,}）...")

            km_full = KMeans(
                n_clusters=self.vocab_size,
                init=km.cluster_centers_,
                n_init=1,
                max_iter=100,
                random_state=self.seed,
                tol=1e-5,
            )
            km_full.fit(X_refine)
            centers = km_full.cluster_centers_
            print(f"[Vocab] Stage 2 完成。Inertia = {km_full.inertia_:.4f} "
                  f"(改善 {(km.inertia_ - km_full.inertia_)/km.inertia_*100:+.2f}%)")
        else:
            centers = km.cluster_centers_

        self._centroids = self._unscale(centers).astype(np.float32)
        self._fix_duplicates(X, min_dist=1e-3)
        self._diagnose()
        return self

    def _fix_duplicates(self, X_scaled: np.ndarray, min_dist: float = 1e-3):
        """
        将近重复 centroid（scaled 空间距离 < min_dist）替换为距所有现有中心最远的数据点。
        保持 vocab_size 不变，消除浪费的槽位。
        """
        C = self._scale(self._centroids)

        # 找出所有重复槽位（保留每对中的第一个）
        dup = np.zeros(len(C), dtype=bool)
        for i in range(len(C)):
            if dup[i]:
                continue
            dists = np.linalg.norm(C[i] - C, axis=1)
            dists[i] = np.inf
            dup |= (dists < min_dist) & ~dup

        n_dup = int(dup.sum())
        if n_dup == 0:
            return

        print(f"[Dedup] 检测到 {n_dup} 个近重复 centroid，用远离现有中心的数据点替换...")

        rng = np.random.default_rng(self.seed)
        sample_idx = rng.choice(len(X_scaled), min(50_000, len(X_scaled)), replace=False)
        X_sample = X_scaled[sample_idx]

        active = C[~dup]                          # 当前非重复中心
        for slot in np.where(dup)[0]:
            # 每个候选点到最近非重复中心的距离
            d = np.linalg.norm(X_sample[:, None] - active[None], axis=-1).min(axis=1)
            best = X_sample[d.argmax()]
            C[slot] = best
            active = np.vstack([active, best])    # 新中心也纳入排斥范围

        self._centroids = self._unscale(C).astype(np.float32)
        print(f"[Dedup] 完成，替换了 {n_dup} 个重复槽位")

    # —— 自动诊断 ——————————————————————————————————————————————————————————

    def _diagnose(self):
        """聚类后质量自检报告。"""
        C = self._centroids
        N = len(C)

        # 1. 近重复检测
        diffs = C[:, None] - C[None]
        dists = np.sqrt((diffs**2).sum(axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_dists = dists.min(axis=1)
        n_dup_strict = int((min_dists < 1e-3).sum())
        n_dup_loose  = int((min_dists < 0.01).sum())

        # 2. 静止 token 占比
        cum_disp = np.sqrt(C[:, 0::3].sum(axis=1)**2 + C[:, 1::3].sum(axis=1)**2)
        # 注：上行用各段累积反推全局位移；保留与 cap_stationary 一致的口径再算一遍
        end_disp = np.sqrt(C[:, 12]**2 + C[:, 13]**2)
        static_rate = float((end_disp < 0.5).mean())

        # 3. 累积转向覆盖
        total_dh = C[:, 2::3].sum(axis=1)

        # 4. 累积位移分位数（按各段位移之和）
        seg_disp = np.sqrt(C[:, 0::3]**2 + C[:, 1::3]**2).sum(axis=1)

        print(f"\n[Diag] ====== 词表质量诊断 ======")
        print(f"  vocab_size = {N}, seg_dim = {SEG_DIM}, angle_weight = {self.angle_weight:.3f}")
        print(f"  近重复 token: {n_dup_strict} (dist<1e-3) / {n_dup_loose} (dist<1e-2)")
        print(f"  静止 token 占比 (末段位移<0.5m): {static_rate*100:.1f}%")
        print(f"  累积位移分位数 (m): "
              f"p10={np.percentile(seg_disp,10):.2f}  "
              f"p50={np.percentile(seg_disp,50):.2f}  "
              f"p90={np.percentile(seg_disp,90):.2f}  "
              f"p99={np.percentile(seg_disp,99):.2f}")
        print(f"  累积转向 (rad): "
              f"min={total_dh.min():.2f} ({np.degrees(total_dh.min()):+.0f}°)  "
              f"max={total_dh.max():.2f} ({np.degrees(total_dh.max()):+.0f}°)")
        print(f"  最近邻间距: mean={min_dists.mean():.4f}  median={np.median(min_dists):.4f}")

        warnings = []
        if n_dup_strict > 0:
            warnings.append(f"⚠ {n_dup_strict} 个严重近重复 token（建议增大 n_init 或检查数据）")
        if static_rate > 0.15:
            warnings.append(f"⚠ 静止率 {static_rate*100:.1f}% 偏高（建议降低 max_stationary_ratio）")
        if static_rate < 0.005 and self.vocab_size >= 512:
            warnings.append(f"⚠ 静止率 {static_rate*100:.1f}% 过低，停车场景可能未被覆盖")

        if warnings:
            for w in warnings:
                print(f"  {w}")
        else:
            print(f"  ✓ 未检测到明显异常")
        print(f"[Diag] ============================\n")

    # —— 编码 ——————————————————————————————————————————————————————————————

    def encode(self, segs: np.ndarray, chunk: int = 8192) -> np.ndarray:
        """segs : (N, 15) → token IDs (N,)，含 N_SPECIAL 偏移。分块避免 OOM。"""
        self._check()
        cs = self._scale(self._centroids)
        out = np.empty(len(segs), dtype=np.int64)
        for i in range(0, len(segs), chunk):
            X = self._scale(segs[i:i + chunk])
            diff = X[:, None] - cs[None]
            out[i:i + chunk] = np.argmin(np.linalg.norm(diff, axis=-1), axis=-1)
        return out + self.N_SPECIAL

    def encode_topk(self, segs: np.ndarray, k: int, chunk: int = 8192) -> np.ndarray:
        """segs : (N, 15) → top-k token IDs (N, k)，含 N_SPECIAL 偏移。"""
        self._check()
        cs = self._scale(self._centroids)
        out = np.empty((len(segs), k), dtype=np.int64)
        for i in range(0, len(segs), chunk):
            X = self._scale(segs[i:i + chunk])
            diff = X[:, None] - cs[None]
            dists = np.linalg.norm(diff, axis=-1)
            out[i:i + chunk] = np.argsort(dists, axis=-1)[:, :k]
        return out + self.N_SPECIAL

    # —— I/O ——————————————————————————————————————————————————————————————

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
        print(f"[Vocab] 已加载 {v.vocab_size}-token 词表（{SEG_DIM}D）← {path}")
        return v

    # —— 内部工具 ——————————————————————————————————————————————————————————

    @property
    def centroids(self) -> np.ndarray:
        self._check()
        return self._centroids

    def _scale(self, x: np.ndarray) -> np.ndarray:
        s = x.copy().astype(np.float32)
        s[:, 2::3] *= float(self.angle_weight)
        return s

    def _unscale(self, x: np.ndarray) -> np.ndarray:
        s = x.copy()
        s[:, 2::3] /= float(self.angle_weight)
        return s

    def _check(self):
        if self._centroids is None:
            raise RuntimeError("词表未初始化，请先调用 fit() 或 load()")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_angle_weight(v: str) -> Union[float, str]:
    if v.lower() == 'auto':
        return 'auto'
    return float(v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="构建运动词表 v3（15D 子轨迹，两段式 KMeans + 分层采样）"
    )
    parser.add_argument("--npz_dir",    required=True)
    parser.add_argument("--save",       default="./vocab/vocab_512.npz")
    parser.add_argument("--vocab_size", type=int,   default=512)
    parser.add_argument("--max_files",  type=int,   default=None)
    parser.add_argument("--batch_size", type=int,   default=4096)
    parser.add_argument("--max_iter",   type=int,   default=300)
    parser.add_argument("--n_init",     type=int,   default=10,  help="MiniBatchKMeans 重启次数（v2 默认 3）")
    parser.add_argument("--no_refine",  action="store_true",  help="关闭 Stage 2 精炼（仅用 MiniBatchKMeans）")
    parser.add_argument("--refine_max_samples", type=int, default=200_000,  help="Stage 2 精炼时的最大样本数")
    parser.add_argument("--angle_weight", type=_parse_angle_weight, default=3.0,  help="角度权重，可填浮点数或 'auto'（按 std 自适应）")
    parser.add_argument("--max_stationary_ratio", type=float, default=0.15,  help="静止 segment 最大占比，默认 0.15")
    parser.add_argument("--stratify", action="store_true",  help="启用分层采样平衡速度分布（推荐 nbr 词表）")
    parser.add_argument("--n_buckets", type=int, default=8,  help="分层采样桶数")
    parser.add_argument("--source", choices=['all', 'ego', 'nbr'], default='all')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    segments = collect_segments(
        args.npz_dir, args.max_files,
        max_stationary_ratio = args.max_stationary_ratio,
        stratify             = args.stratify,
        n_buckets            = args.n_buckets,
        source               = args.source,
    )

    vocab = MotionVocabulary(
        vocab_size   = args.vocab_size,
        angle_weight = args.angle_weight,
        seed         = args.seed,
    )
    vocab.fit(
        segments,
        batch_size         = args.batch_size,
        max_iter           = args.max_iter,
        n_init             = args.n_init,
        refine             = not args.no_refine,
        refine_max_samples = args.refine_max_samples,
    )
    vocab.save(args.save)

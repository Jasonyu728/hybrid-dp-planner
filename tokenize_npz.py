"""
tokenize_npz.py
===============
读取 DiffusionPlanner 的 .npz 文件，添加 SMART 风格的 motion token 字段。

处理后每个 .npz 新增两个字段：
  ego_token_ids        : (18,)      int16   [BOS, 16个token, EOS]
  neighbor_token_ids   : (32, 18)   int16   每个agent的token序列（缺失agent全为PAD=0）

用法
----
  # 第一步：确认单文件正确（dry-run）
  python tokenize_npz.py \
      --vocab vocab_512.npz \
      --file  /path/to/cache/scenario.npz \
      --dry_run

  # 第二步：批量处理整个目录
  python tokenize_npz.py \
      --vocab    vocab_512.npz \
      --data_dir /path/to/cache \
      --workers  8
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from vocab_divide_token_v1 import (
    MotionVocabulary,
    K_TOKENS,
    TOKEN_STEP,
    T_FUT,
    N_SPECIAL,
    _wrap_angle,
    _is_valid_nbr_traj,
)

# token 序列总长度：BOS + 16个motion token + EOS = 18
SEQ_LEN = K_TOKENS + 2

# v5 词表的 centroid 维度
_V5_SEG_DIM = 15
_WORKER_VOCAB = None
_WORKER_NBR_VOCAB = None


def _is_v5_vocab(vocab) -> bool:
    """vocab.centroids.shape[1] == 15 表示 v5 词表（15维子轨迹 token）。"""
    return vocab.centroids.shape[1] == _V5_SEG_DIM


# ── v5：单条轨迹 tokenization（15维查询）────────────────────────────────

def tokenize_future_traj_v5(
    future_traj: np.ndarray,
    vocab,
) -> np.ndarray:
    """
    v5 词表专用 tokenization。

    与 v3/v4 的区别
    ---------------
    - 查询向量：3维终点位移 → 15维子轨迹（5帧全部位移）
    - 参考点更新：同样使用 chosen token 的最后一帧（centroid 索引 [12,13,14]）

    坐标系
    ------
    对 token i，参考点 (x_ref, y_ref, h_ref)，5帧局部位移：
      frame j: dx_j, dy_j = 在 h_ref 朝向下，第 j 帧相对参考点的位移
               dh_j       = 第 j 帧相对参考点的朝向差
    """
    T = len(future_traj)
    if T < TOKEN_STEP or np.allclose(future_traj[:TOKEN_STEP, :2], 0):
        return np.full(SEQ_LEN, vocab.PAD_IDX, dtype=np.int16)

    ids = [vocab.BOS_IDX]
    x_ref, y_ref, h_ref = 0.0, 0.0, 0.0

    # 上层（tokenize_file）已经用 _is_valid_nbr_traj 把 nbr 截到末段有效帧，
    # ego_future 永远是完整 80 帧，所以这里只用 end_idx 越界做边界，不再做
    # mid-block all-zero early-break——与 _tokenize_v5_batch 的截断口径对齐。
    for i in range(K_TOKENS):
        end_idx = (i + 1) * TOKEN_STEP - 1
        if end_idx >= T:
            break

        # ── 构建 15 维查询向量 ──────────────────────────────────────────
        cos_h, sin_h = np.cos(h_ref), np.sin(h_ref)
        sub = []
        for j in range(TOKEN_STEP):
            frame_idx = i * TOKEN_STEP + j
            x_f, y_f, h_f = future_traj[frame_idx]
            dx_g = x_f - x_ref
            dy_g = y_f - y_ref
            dx_l =  cos_h * dx_g + sin_h * dy_g
            dy_l = -sin_h * dx_g + cos_h * dy_g
            dh   = float(_wrap_angle(np.array([h_f - h_ref]))[0])
            sub.extend([dx_l, dy_l, dh])

        seg = np.array([sub], dtype=np.float32)   # (1, 15)

        chosen = int(vocab.encode(seg)[0])

        ids.append(chosen)

        # ── 用 chosen token 最后一帧更新 rolling 参考点 ─────────────────
        # centroid = [dx0,dy0,dh0, ..., dx4,dy4,dh4]，最后一帧在 [12,13,14]
        raw = chosen - vocab.N_SPECIAL
        dx_c = float(vocab.centroids[raw][12])
        dy_c = float(vocab.centroids[raw][13])
        dh_c = float(vocab.centroids[raw][14])
        x_ref = x_ref + cos_h * dx_c - sin_h * dy_c
        y_ref = y_ref + sin_h * dx_c + cos_h * dy_c
        h_ref = float(_wrap_angle(np.array([h_ref + dh_c]))[0])

    while len(ids) - 1 < K_TOKENS:
        ids.append(vocab.PAD_IDX)
    ids.append(vocab.EOS_IDX)
    return np.array(ids, dtype=np.int16)


# ── 核心：单条轨迹的 rolling-match tokenization ───────────────────────────

def tokenize_future_traj(
    future_traj: np.ndarray,          # (80, 3)  [x, y, heading]
    vocab: MotionVocabulary,
) -> np.ndarray:
    """
    将一条未来轨迹转换为 token 序列（含 BOS/EOS）。

    Rolling matching 逻辑
    ---------------------
    - 每步以**上一个 token 解码出的位置**为参考点，而不是 GT 位置
    - 这模拟了推理时的自回归行为，防止 compounding error
    Returns
    -------
    token_ids : (SEQ_LEN,) = (18,)  int16
                [BOS, t1, t2, ..., t16, EOS]
                缺失/无效轨迹返回全 PAD 序列
    """
    T = len(future_traj)
    if T < TOKEN_STEP or np.allclose(future_traj[:TOKEN_STEP, :2], 0):
        return np.full(SEQ_LEN, vocab.PAD_IDX, dtype=np.int16)

    ids = [vocab.BOS_IDX]

    # rolling 参考状态：从 ego 原点 (0,0,0) 开始
    x_ref, y_ref, h_ref = 0.0, 0.0, 0.0

    for i in range(K_TOKENS):
        end_idx = (i + 1) * TOKEN_STEP - 1
        if end_idx >= T:
            break
        block = future_traj[i * TOKEN_STEP:(i + 1) * TOKEN_STEP]
        if np.allclose(block[:, :2], 0):
            break

        x_end, y_end, h_end = future_traj[end_idx]

        # 计算从 rolling 参考点到 GT 终点的局部位移
        dx_g = x_end - x_ref
        dy_g = y_end - y_ref
        cos_h, sin_h = np.cos(h_ref), np.sin(h_ref)
        dx_l =  cos_h * dx_g + sin_h * dy_g
        dy_l = -sin_h * dx_g + cos_h * dy_g
        dh   = float(_wrap_angle(np.array([h_end - h_ref]))[0])

        seg = np.array([[dx_l, dy_l, dh]], dtype=np.float32)

        chosen = int(vocab.encode(seg)[0])

        ids.append(chosen)

        # 用**所选 token** 的 centroid 更新 rolling 参考点
        raw = chosen - vocab.N_SPECIAL
        dx_c, dy_c, dh_c = vocab.centroids[raw]
        cos_h, sin_h = np.cos(h_ref), np.sin(h_ref)
        x_ref = x_ref + cos_h * dx_c - sin_h * dy_c
        y_ref = y_ref + sin_h * dx_c + cos_h * dy_c
        h_ref = float(_wrap_angle(np.array([h_ref + dh_c]))[0])

    # 不足 K_TOKENS 时用 PAD 补齐
    while len(ids) - 1 < K_TOKENS:
        ids.append(vocab.PAD_IDX)

    ids.append(vocab.EOS_IDX)
    return np.array(ids, dtype=np.int16)  # (18,)


# ── 单文件处理 ────────────────────────────────────────────────────────────

def tokenize_file(
    fp: Path,
    vocab: MotionVocabulary,
    dry_run: bool = False,
    nbr_vocab: Optional[MotionVocabulary] = None,
) -> bool:
    """
    对单个 .npz 文件执行 tokenization，并将结果写回同一文件。

    新增字段
    --------
    ego_token_ids          : (18,)     int16
    neighbor_token_ids     : (32, 18)  int16

    参数
    ----
    vocab     : ego 词表（也作为 neighbor 的后备词表）
    nbr_vocab : neighbor 专用词表（可选）；若提供则 neighbor 使用该词表
    """
    try:
        data = dict(np.load(fp, allow_pickle=False))
    except Exception as e:
        print(f"  [ERROR] 无法加载 {fp.name}: {e}")
        return False

    # neighbor 使用专用词表（若有），否则退回到 ego 词表
    _nbr_vocab = nbr_vocab if nbr_vocab is not None else vocab

    # 自动选择 tokenization 函数（v5 词表用 15 维查询，旧词表用 3 维）
    _tok_fn_ego = tokenize_future_traj_v5 if _is_v5_vocab(vocab) else tokenize_future_traj
    _tok_fn_nbr = tokenize_future_traj_v5 if _is_v5_vocab(_nbr_vocab) else tokenize_future_traj

    # ── ego ──────────────────────────────────────────────────────────────
    ego_tok = None
    if "ego_agent_future" in data:
        ego_fut = data["ego_agent_future"]   # (80, 3)
        ego_tok = _tok_fn_ego(ego_fut, vocab)

    # ── neighbors ────────────────────────────────────────────────────────
    nbr_tok = None
    if "neighbor_agents_future" in data:
        nbr_fut = data["neighbor_agents_future"]   # (32, 80, 3)
        N = nbr_fut.shape[0]
        nbr_tok = np.zeros((N, SEQ_LEN), dtype=np.int16)

        for n in range(N):
            ok, end = _is_valid_nbr_traj(nbr_fut[n])
            if not ok:
                continue
            nbr_traj = nbr_fut[n, :end]
            nbr_tok[n] = _tok_fn_nbr(nbr_traj, _nbr_vocab)

    # ── 打印（dry_run）或写回 ─────────────────────────────────────────────
    if dry_run:
        print(f"[DRY RUN] {fp.name}")
        if ego_tok is not None:
            print(f"  ego_token_ids     : {ego_tok}")
        if nbr_tok is not None:
            valid = [(i, nbr_tok[i]) for i in range(nbr_tok.shape[0])
                     if nbr_tok[i, 0] != vocab.PAD_IDX]
            print(f"  有效 neighbor 数  : {len(valid)}/{nbr_tok.shape[0]}")
            for i, t in valid[:3]:
                print(f"    neighbor[{i}]: {t}")
        return True

    # 写入新字段
    if ego_tok     is not None: data["ego_token_ids"]              = ego_tok
    if nbr_tok     is not None: data["neighbor_token_ids"]         = nbr_tok

    # 原子写入：先写到 .npz 以外的临时路径，再重命名
    # 注意：np.savez_compressed 若路径不含 .npz 后缀会自动追加，
    # 所以用不含 .npz 的 stem 作为临时前缀
    tmp_stem = str(fp.with_suffix("")) + "_tmp_"
    tmp_final = tmp_stem + ".npz"
    try:
        np.savez_compressed(tmp_stem, **data)   # 实际写出 tmp_stem + ".npz"
        os.replace(tmp_final, fp)
    except Exception as e:
        print(f"  [ERROR] 写入失败 {fp.name}: {e}")
        if os.path.exists(tmp_final):
            os.remove(tmp_final)
        return False

    return True


# ── 批量处理目录 ──────────────────────────────────────────────────────────

def tokenize_directory(
    data_dir: str,
    vocab_path: str,
    num_workers: int = 1,
    dry_run: bool = False,
    skip_existing: bool = True,
    nbr_vocab_path: Optional[str] = None,
) -> None:
    vocab = MotionVocabulary.load(vocab_path)
    nbr_vocab = MotionVocabulary.load(nbr_vocab_path) if nbr_vocab_path else None
    if nbr_vocab is not None:
        print(f"[Tokenize] 使用独立 neighbor 词表：{nbr_vocab_path}")
    else:
        print(f"[Tokenize] ego/neighbor 共用词表：{vocab_path}")

    files = sorted(Path(data_dir).rglob("*.npz"))
    files = [f for f in files if not f.suffix == ".tmp"]

    print(f"[Tokenize] 发现 {len(files)} 个 .npz 文件")

    if skip_existing and not dry_run:
        to_do = []
        for fp in files:
            try:
                if "ego_token_ids" not in np.load(fp, allow_pickle=False).files:
                    to_do.append(fp)
            except Exception:
                to_do.append(fp)
        print(f"[Tokenize] 需要处理 {len(to_do)} 个（{len(files)-len(to_do)} 个已完成）")
        files = to_do

    if num_workers <= 1:
        _run_single(files, vocab, dry_run, nbr_vocab)
    else:
        _run_parallel(files, vocab_path, num_workers, nbr_vocab_path)


def _run_single(files, vocab, dry_run, nbr_vocab=None):
    ok = err = 0
    for i, fp in enumerate(files):
        if i % 1000 == 0:
            print(f"  {i}/{len(files)}  ok={ok}  err={err}")
        if tokenize_file(fp, vocab, dry_run=dry_run, nbr_vocab=nbr_vocab):
            ok += 1
        else:
            err += 1
    print(f"[Tokenize] 完成。ok={ok}  err={err}")


def _init_worker(vocab_path, nbr_vocab_path):
    global _WORKER_VOCAB, _WORKER_NBR_VOCAB
    _WORKER_VOCAB = MotionVocabulary.load(vocab_path)
    _WORKER_NBR_VOCAB = MotionVocabulary.load(nbr_vocab_path) if nbr_vocab_path else None


def _worker_fn(fp_str):
    return tokenize_file(Path(fp_str), _WORKER_VOCAB, nbr_vocab=_WORKER_NBR_VOCAB)


def _run_parallel(files, vocab_path, n_workers, nbr_vocab_path=None):
    from multiprocessing import Pool
    args = [str(f) for f in files]
    ok = err = 0
    with Pool(n_workers, initializer=_init_worker, initargs=(vocab_path, nbr_vocab_path)) as pool:
        for i, success in enumerate(pool.imap_unordered(_worker_fn, args, chunksize=64)):
            if i % 1000 == 0:
                print(f"  {i}/{len(files)}")
            if success: ok += 1
            else: err += 1
    print(f"[Tokenize] 完成（并行）。ok={ok}  err={err}")


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab",     required=True,
                        help="ego 词表路径（同时也作为 neighbor 的后备词表）")
    parser.add_argument("--nbr_vocab", default=None,
                        help="neighbor 专用词表路径（可选；不指定则与 ego 共用词表）")
    parser.add_argument("--data_dir",  default=None,  help="npz 文件目录（批量模式）")
    parser.add_argument("--file",      default=None,  help="单个 npz 文件（测试模式）")
    parser.add_argument("--workers",   type=int, default=8)
    parser.add_argument("--dry_run",   action="store_true", help="只打印不写入")
    parser.add_argument("--no_skip",   action="store_true", help="重新处理已有 token 的文件")
    args = parser.parse_args()

    if args.file:
        vocab     = MotionVocabulary.load(args.vocab)
        nbr_vocab = MotionVocabulary.load(args.nbr_vocab) if args.nbr_vocab else None
        tokenize_file(Path(args.file), vocab, dry_run=args.dry_run, nbr_vocab=nbr_vocab)
    elif args.data_dir:
        tokenize_directory(
            args.data_dir, args.vocab,
            num_workers=args.workers,
            dry_run=args.dry_run,
            skip_existing=not args.no_skip,
            nbr_vocab_path=args.nbr_vocab,
        )
    else:
        parser.error("请指定 --data_dir 或 --file")

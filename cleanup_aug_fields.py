"""
cleanup_aug_fields.py
=====================
清除 npz 文件里残留的 *_aug token 字段。

背景：早期 tokenize_npz.py 在 --augment 模式下会给每个 npz 写入
ego_token_ids_aug / neighbor_token_ids_aug 两个字段。新版代码已经
不再读取这些字段（数据增强改由 train_epoch.py 在线做），但老 npz
里还存着它们，占空间。

特性：
- 并行处理，按 chunksize 分配任务
- 原子写入：先写临时文件，成功再 rename，中断不会留下半成品
- skip_existing：再次跑可以跳过已清理的文件
- dry-run 模式：先看能省多少空间，不实际修改

用法：
  # 先看一眼能省多少
  python cleanup_aug_fields.py --data_dir /path/to/npz/dir --dry_run

  # 实际清理（建议先用 --max_files 200 试一小批）
  python cleanup_aug_fields.py --data_dir /path/to/npz/dir --workers 16

  # 清理 + 只跑 200 个测试
  python cleanup_aug_fields.py --data_dir /path/to/npz/dir --max_files 200
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

AUG_KEYS = ("ego_token_ids_aug", "neighbor_token_ids_aug")


def _has_aug(fp: Path) -> Tuple[bool, int]:
    """检查文件是否含 _aug 字段，返回 (是否含, 这些字段总字节数)。"""
    try:
        with np.load(fp, allow_pickle=False) as data:
            has = False
            nbytes = 0
            for k in AUG_KEYS:
                if k in data.files:
                    has = True
                    nbytes += data[k].nbytes
        return has, nbytes
    except Exception:
        return False, 0


def _strip_aug_one(fp_str: str) -> Tuple[str, bool, str]:
    """
    去掉一个 npz 文件里的 _aug 字段。
    返回 (路径, 是否修改了, 错误信息)。
    """
    fp = Path(fp_str)
    try:
        with np.load(fp, allow_pickle=False) as data:
            keys = set(data.files)
            if not (keys & set(AUG_KEYS)):
                return fp_str, False, ""   # 没有 _aug，跳过
            new_data = {k: data[k] for k in data.files if k not in AUG_KEYS}

        # 原子写入：先写 tmp，成功再替换
        tmp_stem = str(fp.with_suffix("")) + "_cleanup_tmp"
        tmp_final = tmp_stem + ".npz"
        try:
            np.savez_compressed(tmp_stem, **new_data)
            os.replace(tmp_final, fp)
        except Exception as e:
            if os.path.exists(tmp_final):
                os.remove(tmp_final)
            return fp_str, False, f"写入失败: {e}"
        return fp_str, True, ""
    except Exception as e:
        return fp_str, False, f"读取失败: {e}"


def _run_dry(files: List[Path]) -> None:
    n_with = 0
    bytes_total = 0
    for i, fp in enumerate(files):
        if i % 500 == 0:
            print(f"  scan {i}/{len(files)} ...")
        has, nb = _has_aug(fp)
        if has:
            n_with += 1
            bytes_total += nb

    print()
    print(f"[Dry-run] 总文件数:           {len(files):,}")
    print(f"[Dry-run] 含 _aug 字段的文件: {n_with:,}  ({n_with/max(len(files),1)*100:.1f}%)")
    print(f"[Dry-run] _aug 内存总占用:    {bytes_total:,} bytes  ({bytes_total/1e6:.2f} MB)")
    print(f"[Dry-run] 压缩后磁盘节省大约: {bytes_total/4e6:.2f} MB（估算，按 4x 压缩比）")
    print("[Dry-run] 加 --execute 真正清理。")


def _run_serial(files: List[Path]) -> Tuple[int, int, int]:
    n_modified = n_skipped = n_error = 0
    for i, fp in enumerate(files):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(files)}  modified={n_modified} skipped={n_skipped} error={n_error}")
        _, modified, err = _strip_aug_one(str(fp))
        if err:
            n_error += 1
            if n_error < 10:
                print(f"  [ERR] {fp.name}: {err}")
        elif modified:
            n_modified += 1
        else:
            n_skipped += 1
    return n_modified, n_skipped, n_error


def _run_parallel(files: List[Path], n_workers: int) -> Tuple[int, int, int]:
    from multiprocessing import Pool

    n_modified = n_skipped = n_error = 0
    with Pool(n_workers) as pool:
        for i, (fp_str, modified, err) in enumerate(
            pool.imap_unordered(_strip_aug_one, [str(f) for f in files], chunksize=64)
        ):
            if i % 500 == 0 and i > 0:
                print(f"  {i}/{len(files)}  modified={n_modified} skipped={n_skipped} error={n_error}")
            if err:
                n_error += 1
                if n_error < 10:
                    print(f"  [ERR] {fp_str}: {err}")
            elif modified:
                n_modified += 1
            else:
                n_skipped += 1
    return n_modified, n_skipped, n_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="npz 文件目录（会递归扫描）")
    parser.add_argument("--workers", type=int, default=8, help="并行 worker 数")
    parser.add_argument("--dry_run", action="store_true", help="只扫描不修改")
    parser.add_argument("--max_files", type=int, default=None, help="只处理前 N 个文件（测试用）")
    args = parser.parse_args()

    root = Path(args.data_dir)
    if not root.is_dir():
        print(f"[ERROR] {root} 不是目录")
        sys.exit(1)

    print(f"[Cleanup] 扫描 {root} ...")
    files = sorted(root.rglob("*.npz"))
    print(f"[Cleanup] 找到 {len(files):,} 个 .npz")

    if args.max_files:
        files = files[:args.max_files]
        print(f"[Cleanup] --max_files {args.max_files}，只处理前 {len(files):,} 个")

    if args.dry_run:
        _run_dry(files)
        return

    print(f"[Cleanup] 开始清理（并行 {args.workers} workers）...")
    if args.workers <= 1:
        n_mod, n_skip, n_err = _run_serial(files)
    else:
        n_mod, n_skip, n_err = _run_parallel(files, args.workers)

    print()
    print(f"[Cleanup] 完成。")
    print(f"  修改:   {n_mod:,}  (实际清掉了 _aug 字段)")
    print(f"  跳过:   {n_skip:,}  (已经没有 _aug 字段)")
    print(f"  错误:   {n_err:,}")


if __name__ == "__main__":
    main()

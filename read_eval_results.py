"""
read_eval_results.py
====================
读取 nuPlan 闭环仿真评估结果，只保存论文 Table 2 的 6 个核心指标。

核心指标
--------
  Score       → aggregator final_score.score
  Collisions  → no_ego_at_fault_collisions
  TTC         → time_to_collision_within_bound
  Drivable    → drivable_area_compliance
  Comfort     → ego_is_comfortable
  Progress    → ego_progress_along_expert_route

用法：
    python read_eval_results.py --tag v1_ep480
    python read_eval_results.py --exp_dir <路径> --tag v1_ep480
    python read_eval_results.py --base_dir <路径> --tag v1_ep480
"""

import argparse
import glob
import os
from datetime import datetime, timezone, timedelta

import pandas as pd

# 指标分两组：
#   第一组：论文 Table 2 的 6 个核心指标
#   第二组：参与 Score 计算但论文未单独列出的指标
#     Score = Collisions × Drivable × Direction × Making × mean_weighted(加法项)
#     加法项权重：TTC/Comfort/Progress(w=5), SpeedLimit(w=4)
METRICS = [
    # ── 论文 Table 2 指标 ────────────────────────────
    ("Score",      None),
    ("Collisions", "no_ego_at_fault_collisions"),
    ("TTC",        "time_to_collision_within_bound"),
    ("Drivable",   "drivable_area_compliance"),
    ("Comfort",    "ego_is_comfortable"),
    ("Progress",   "ego_progress_along_expert_route"),
    # ── Score 计算相关（诊断用）─────────────────────
    ("Direction",  "driving_direction_compliance"),   # 乘法项
    ("Making",     "ego_is_making_progress"),         # 乘法项，50%会直接砍半Score
    ("SpeedLimit", "speed_limit_compliance"),         # 加法项 w=4
]

CSV_COLUMNS = ["tag", "timestamp",
               "Score", "Collisions", "TTC", "Drivable", "Comfort", "Progress",
               "Direction", "Making", "SpeedLimit"]


def find_latest_exp_dir(base_dir: str) -> str:
    dirs = glob.glob(os.path.join(base_dir, "**", "aggregator_metric"), recursive=True)
    if not dirs:
        raise FileNotFoundError(f"在 {base_dir} 下找不到 aggregator_metric 目录")
    dirs.sort(key=os.path.getmtime, reverse=True)
    return os.path.dirname(dirs[0])


def read_aggregator(exp_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(exp_dir, "aggregator_metric", "*.parquet"))
    if not files:
        raise FileNotFoundError(f"找不到 aggregator parquet：{exp_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def extract_metrics(agg_df: pd.DataFrame) -> dict:
    """从 aggregator final_score 行提取 6 个核心指标，返回 {列名: 百分制float}。"""
    result = {label: float("nan") for label, _ in METRICS}

    final = agg_df[agg_df["scenario"].astype(str).str.strip() == "final_score"]
    if final.empty:
        return result

    row = final.iloc[0]

    # Score
    if "score" in final.columns and pd.notna(row["score"]):
        result["Score"] = round(float(row["score"]) * 100, 2)

    # 其余 5 个
    for label, col in METRICS[1:]:
        if col in final.columns and pd.notna(row[col]):
            result[label] = round(float(row[col]) * 100, 2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir",    type=str, default=None, help="直接指定评估结果目录")
    parser.add_argument("--base_dir",   type=str,
        default="/data3/yuzhuoyi/AD/DiffusionPlanner/nuplan-devkit/nuplan/dataset/exp/exp",
        help="自动搜索时的根目录")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--tag",        type=str, required=True, help="实验标签，如 v1_ep480")
    args = parser.parse_args()

    # 确定结果目录
    if args.exp_dir:
        exp_dir = args.exp_dir
    else:
        print(f"自动搜索最新结果（{args.base_dir}）...")
        exp_dir = find_latest_exp_dir(args.base_dir)
        print(f"找到：{exp_dir}")

    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")

    # 读取指标
    agg_df  = read_aggregator(exp_dir)
    metrics = extract_metrics(agg_df)

    # 打印
    print("\n" + "="*50)
    print(f"  实验: {args.tag}    时间: {timestamp}")
    print("="*50)
    print(f"  {'指标':<12} {'得分 (%)':>10}")
    print(f"  {'-'*12} {'-'*10}")
    for label, _ in METRICS:
        v = metrics[label]
        v_str = f"{v:10.2f}" if v == v else f"{'NaN':>10}"
        print(f"  {label:<12} {v_str}")
    print("="*50)

    # 保存到 CSV（只有 6 个指标 + tag + timestamp）
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "all_results.csv")

    new_row = pd.DataFrame([{
        "tag":       args.tag,
        "timestamp": timestamp,
        **metrics,
    }], columns=CSV_COLUMNS)

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        # 保证列一致
        for col in CSV_COLUMNS:
            if col not in existing.columns:
                existing[col] = float("nan")
        combined = pd.concat([existing[CSV_COLUMNS], new_row], ignore_index=True)
    else:
        combined = new_row

    combined.to_csv(csv_path, index=False)
    print(f"\n结果已追加到：{csv_path}")


if __name__ == "__main__":
    main()

# CHANGELOG

## [v11] 2026-04-28

- `diffusion_planner/utils/diff_decode.py`: 新增，将 `_differentiable_decode` 提取为共享工具函数（training + guidance 两处复用）。
- `diffusion_planner/train_epoch.py`: 将内联的 `_differentiable_decode` 替换为从 `utils.diff_decode` 的 import。
- `diffusion_planner/model/guidance/token_collision.py`: 新增，实现 token embedding 空间的碰撞 guidance 函数（SAT 矩形碰撞检测，梯度路径通过 `_differentiable_decode` 回传到 ego embedding）。
- `diffusion_planner/model/guidance/guidance_wrapper.py`: 新增 `TokenGuidanceWrapper`（token 版 guidance wrapper，使用 x0 预测值替换 xt 做几何计算，同时保留 xt 的梯度路径）。
- `diffusion_planner/model/module/decoder.py`: 推理路径 `classifier_kwargs` 新增 `ego_emb_w`、`nbr_emb_w`、`ego_centroids`、`nbr_centroids`，供 `token_collision_guidance_fn` 调用。
- `diffusion_planner/config/planner/diffusion_planner_token_guidance.yaml`: 新增，planner 配置文件，`guidance_fn` 指向 `TokenGuidanceWrapper`。
- `sim_guidance_demo.sh`: `PLANNER` 从 `diffusion_planner_guidance` 改为 `diffusion_planner_token_guidance`。

---

## [v10] 2026-04-28

- `diffusion_planner/train_epoch.py`: 移除 `loss_smooth`、`loss_route`、`loss_col` 及其统计变量，loss 恢复为 diffusion + commitment(0.25) + reconstruction(1.0) 三项干净版本。

---

## [v9] 2026-04-28

### `diffusion_planner/train_epoch.py`

**`loss_route` 权重 0.1 → 0.3**
- 针对 Drivable 25% 问题加强路线约束，server 2（128维）专用配置。

**新增 `loss_col`（collision avoidance loss），权重 0.05**
- 动机：Collisions 8.33%、TTC 8.33%，route loss 只管在路上，不管撞不撞邻居。
- 实现：用可微分重建轨迹 `traj_rec[:, :, :2]` 与 `batch['neighbor_agents_future'][:, :N_pred, :, :2]`（邻居 GT 未来 xy）计算逐帧距离，低于安全距离 3m 的部分产生惩罚：
  ```python
  nbr_dists = (ego_xy_e - nbr_xy).norm(dim=-1)   # (B, N, 80)
  risk      = (3.0 - nbr_dists).clamp(min=0) * nbr_valid.unsqueeze(-1)
  loss_col  = risk.mean()
  ```
- 权重 0.05 保守，避免早期训练梯度过大（risk 量级 0–3m）。
- 新增 `sum_loss_col` 统计，loss_dict 新增 `collision_avoidance_loss` 键。

---

## [v8] 2026-04-28

### `diffusion_planner/utils/token_trajectory_decoder.py`

**推理平滑 sigma：1.0 → 2.0**
- 高斯核 sigma 从 1.0 加大到 2.0，对 token 边界处的速度跳变施加更强平滑。
- 纯推理侧改动，不需要重新训练，直接用现有 checkpoint 重跑评估即可观察 Comfort 变化。

---

## [v7] 2026-04-28

### `diffusion_planner/train_epoch.py`

**移除 `time_w` 前 4 token 偏置**
- 删除 `time_w[:4] = 2.0`，改为全 1 均匀权重。
- 原因：800 eps 评估发现 Making 升至 100% 但 Collisions 跌至 8.33%，模型被迫在所有场景前 2s 预测高运动 token，遇到障碍物或红灯也强行前进导致碰撞。

**新增 `loss_route`（route following loss），权重 0.1**
- 动机：ego 出现乱开现象，Drivable 仅 25%。训练 loss 全在 embedding 空间，没有任何路线跟随信号。
- 实现：将 `traj_rec[:, :, :2]`（可微分重建轨迹的 xy）与 `batch['route_lanes'][..., :2]` 展开的 500 个路线点做 `torch.cdist`，取每帧最近点距离后 `clamp(max=10.0).mean()`：
  ```python
  route_pts  = batch['route_lanes'][..., :2].reshape(B, -1, 2)   # (B, 500, 2)
  route_valid = route_pts.abs().sum(-1) > 1e-3
  dists      = torch.cdist(traj_rec[:, :, :2], route_pts)        # (B, 80, 500)
  dists      += (~route_valid).unsqueeze(1).float() * 1e6
  loss_route = dists.min(dim=-1).values.clamp(max=10.0).mean()
  ```
- clamp(max=10.0) 防止早期训练轨迹严重偏离时梯度爆炸。
- 梯度链路：`loss_route → traj_rec → x0_ego → ego_token_emb`，同时约束 embedding 和 DiT。
- 新增 `sum_loss_route` 统计，loss_dict 新增 `route_following_loss` 键。

### `vocab_divide_token_v1.py`

**新增 `_fix_duplicates` 方法**
- 动机：n_init=30 时诊断仍显示 18 个近重复 token，增加 n_init 边际收益递减。
- 实现：Stage 2 KMeans 完成后，在 scaled 空间中找出距离 < 1e-3 的重复 centroid 对，将重复槽位替换为距所有现有中心最远的数据点（从 50k 采样点中选）。新中心纳入排斥范围后依次处理，保证替换后互不重复。
- `fit()` 中自动调用，`_diagnose()` 前执行，最终报告中近重复数应降为 0。

### `.claude/commands/log.md`

**更新 /log skill**：在更新 CHANGELOG 后自动 stage 相关文件并创建 git commit 快照，方便精确回退。

---

## [v6] 2026-04-27

### `diffusion_planner/train_epoch.py`

**新增 `loss_smooth`（smoothness loss），权重 0.1**

- **动机**：nuPlan Comfort 指标仅 8.33%，根本原因是 token 边界处速度不连续，导致加速度（jerk）超出舒适阈值。每个 token 在 decode 时是独立的局部运动段，段与段之间速度跳变无法通过 reconstruction loss 消除。
- **实现**：在 `_differentiable_decode` 得到可微分重建轨迹 `traj_rec (B, 80, 3)` 后，对 xy 坐标的二阶有限差分施加 L2 惩罚：
  ```python
  vel         = traj_rec[:, 1:, :2] - traj_rec[:, :-1, :2]   # (B, 79, 2) 速度
  acc         = vel[:, 1:] - vel[:, :-1]                      # (B, 78, 2) 加速度
  loss_smooth = (acc ** 2).mean()
  ```
- **效果**：训练时模型被迫选择加速度小的 token 序列组合，边界处速度跳变受到惩罚。
- **数值量级**：匀速场景 loss ≈ 0；token 边界跳变 0.5 m/s 时 loss ≈ 0.0025；0.1 权重不会淹没其余 loss 项。
- **统计**：新增 `sum_loss_smooth` 累加器，loss_dict 新增 `smoothness_loss` 键。
- **完整 loss**：`alpha * loss_ego + loss_nbr + 0.25 * loss_commit + 1.0 * loss_recon + 0.1 * loss_smooth`

---

## [v5] 2026-04-27
- `train_epoch.py`: 新增 `_differentiable_decode`（soft NN → soft centroid → 滚动解码全局轨迹）和 `loss_recon`（与真实未来轨迹对比，权重 1.0），防止可训练 codebook 塌缩到 0；保存增强后的 `ego_future_gt` 用于重建监督；统计新增 `reconstruction_loss`

## [v4] 2026-04-27
- `train_predictor.py`: `find_unused_parameters=True` → `False`，新增 `_set_static_graph()`，修复 DDP 训练时 "Parameter marked as ready twice" 崩溃

## [v3] 2026-04-27
- `decoder.py`: embedding 表改为可训练（`requires_grad=True`），`preproj` hidden_features 改为 `max(512, output_dim//2)`
- `train_epoch.py`: 移除 embedding 查表的 `no_grad`，加噪前对 x0 做 `detach`，loss 拆分为 diffusion loss + commitment loss（0.25 权重）
- `train_predictor.py`: 删除 `--guidance_scale` 参数

## [v2] 2026-04-27
- `train_epoch.py`: ego loss 加入场景级权重（按轨迹位移，上限 2.0）和时序权重（前 4 个 token ×2）

## [v1] 2026-04-26
- `vocab_divide_token_v1_kmedoids.py`: 新增，基于 `vocab_divide_token_v1.py`，将 Stage 2 KMeans 替换为 KMedoids，`init='k-medoids++'`

## [v0] 基线
- `decoder.py`: 新增 `ego_token_emb` / `nbr_token_emb`（`nn.Embedding`）和 `ego_traj_decoder` / `nbr_traj_decoder`
- `train_epoch.py`: token ID → embedding 空间扩散训练（原版为连续轨迹）
- `token_trajectory_decoder.py`: 新增，token ID 序列解码为连续轨迹

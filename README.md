# Hybrid-DP-Planner 项目总结

最后更新：2026-05-13

本文档用于记录当前仓库的整体方案、代码结构、训练流程、仿真流程、关键设计选择、历史问题和后续优化方向。它面向后续继续开发、复现实验、迁移服务器环境、以及向他人解释项目时使用。

## 1. 项目定位

本项目基于原始 Diffusion Planner 自动驾驶规划模型进行改造，目标是在 nuPlan closed-loop simulation（闭环仿真）中获得更稳定、更高分的 ego trajectory planning（自车轨迹规划）结果。

原始 Diffusion Planner 的核心思想是：输入历史车辆状态、地图、路线等场景条件，通过 Encoder 编码上下文，再用 DiT diffusion model（Diffusion Transformer，扩散 Transformer）生成未来 8 秒轨迹。

当前项目的核心改造方向是：引入 Discrete Motion Tokenization（离散运动 token 化），把连续未来轨迹转换为 motion token（运动原语）序列，再在 token embedding latent space（token 嵌入向量空间）中训练 diffusion model，同时用 continuous head（连续轨迹输出头）直接输出 80 帧连续轨迹用于 sim。

当前最重要的结论是：

```text
不要在 sim 中依赖 hard token decode 作为最终输出。
当前推荐方案是 hybrid continuous:
DiT -> x0_hat latent -> continuous_head -> 80 帧连续轨迹。
token classifier 只作为辅助监督，不作为最终 sim 输出。
```

## 2. 与原始 Diffusion Planner 的关系

原始 Diffusion Planner：

```text
scene inputs -> Encoder -> DiT diffusion -> continuous trajectory
```

当前项目：

```text
scene inputs -> Encoder -> DiT diffusion in token latent space -> x0_hat
                                                       |
                                                       +-> continuous_head -> continuous trajectory for sim
                                                       |
                                                       +-> classifier -> token IDs for auxiliary CE loss
```

主要差异：

| 维度 | 原始 Diffusion Planner | 当前 Hybrid Token 方案 |
|---|---|---|
| 扩散空间 | 连续轨迹空间 | token embedding latent space |
| 训练目标 | 未来连续轨迹 | token latent MSE + token CE + continuous trajectory SmoothL1 |
| sim 输出 | 连续轨迹 | continuous_head 输出的连续轨迹 |
| token 用途 | 无 | 结构化 latent 表示和辅助监督 |
| hard token decode | 无 | 保留为实验路径，但不是当前推荐 sim 路径 |

## 3. 当前核心方案概述

当前方案可以拆成五个阶段。

### 3.1 构建 motion vocabulary（运动词表）

使用训练集中的未来轨迹构建 motion token vocabulary（运动 token 词表）。

处理逻辑：

```text
future trajectory: 80 frames, 8s, 10Hz
每 5 帧切成一个 token block
80 / 5 = 16 tokens
每个 token 表示 0.5s 局部运动片段
```

当前主线使用 v5 token 格式：

```text
一个 token centroid = 15 维
15 = 5 frames * 3
每帧 3 维: dx, dy, dh
```

其中：

| 字段 | 含义 |
|---|---|
| `dx` | 相对 rolling reference 的局部 x 位移 |
| `dy` | 相对 rolling reference 的局部 y 位移 |
| `dh` | 相对 rolling reference 的 heading 变化 |

rolling reference（滚动参考点）指：每个 token 的局部坐标系不是直接用 GT 上一个点，而是用上一个已选 token centroid 解码出的状态作为参考。这更接近推理时 token 连续拼接的实际行为。

相关文件：

| 文件 | 作用 |
|---|---|
| `vocab_divide_token_v1.py` | 构建 ego/nbr motion vocabulary |
| `vocab.py` | 词表构建或分析辅助脚本 |
| `vocab_kmedoids.py` | k-medoids 版本的词表实验脚本 |
| `vocabulary_v1.py` | motion vocabulary API，早期/基础版本 |
| `vocabulary_v2.py` | motion vocabulary API，支持 v5 15D token |
| `vocab/` | 小型词表文件，已提交到仓库用于复现和分析 |

### 3.2 对 npz 数据做 tokenize

原始数据来自 `data_process.py` 生成的 `.npz` 文件。每个 `.npz` 包含：

```text
ego_current_state
ego_agent_future
neighbor_agents_past
neighbor_agents_future
lanes
route_lanes
static_objects
...
```

`tokenize_npz.py` 会读取每个 `.npz`，根据词表新增：

```text
ego_token_ids: shape (18,)
neighbor_token_ids: shape (32, 18)
```

为什么是 18：

```text
18 = BOS + 16 motion tokens + EOS
```

特殊 token：

| Token | ID | 含义 |
|---|---:|---|
| PAD | 0 | padding，无效 agent 或无效 token |
| BOS | 1 | begin of sequence，序列开始 |
| EOS | 2 | end of sequence，序列结束 |
| motion tokens | >= 3 | 实际运动 token |

相关文件：

| 文件 | 作用 |
|---|---|
| `tokenize_npz.py` | 离线 tokenization 主脚本 |
| `cleanup_aug_fields.py` | 清理旧的增强字段，避免数据字段污染 |
| `diffusion_planner/utils/dataset.py` | 训练时读取 token 字段 |

### 3.3 构造 token embedding latent

每个 motion token ID 会查表得到一个 embedding vector（嵌入向量）。

当前默认参数：

```text
token_emb_dim = 64
k_tokens = 16
latent_dim per agent = 16 * 64 = 1024
```

也就是说，每个 agent 的扩散状态不是 80 帧轨迹，而是一个 1024 维 latent。

embedding 初始化方式：

```text
centroid: 15D
random projection: 15 x D
motion embedding = centroid @ projection
normalize to roughly N(0, 1)
```

当前推荐配置：

```text
learnable_token_emb = False
```

含义是 embedding table 固定，不参与训练。这样更稳定，也避免 embedding 自己漂移导致 token 几何语义被破坏。

如果未来打开：

```text
learnable_token_emb = True
lambda_emb_commit > 0
```

那么需要 commitment loss（承诺损失）约束 embedding，否则 embedding 可能变成任意可训练参数，不再对应运动词表的物理含义。

相关文件：

| 文件 | 作用 |
|---|---|
| `diffusion_planner/model/module/decoder.py` | 构建 ego/nbr embedding table |
| `train_predictor.py` | 定义 `--token_emb_dim`，默认 64 |

### 3.4 DiT 在 token latent space 中扩散

训练时流程如下：

```text
GT token IDs
    -> embedding lookup
    -> x0: [B, P, 16 * D]
    -> VPSDE 加噪
    -> xt
    -> DiT(xt, scene condition, diffusion time)
    -> pred / x0_hat
```

符号说明：

| 符号 | 含义 |
|---|---|
| `x0` | 由 GT token IDs 查 embedding 得到的干净 latent |
| `xt` | diffusion 加噪后的 latent |
| `pred` | DiT 直接输出 |
| `x0_hat` | 从 `pred` 恢复出的干净 latent 估计 |
| `target` | 根据 diffusion model type 生成的训练目标 |

当前默认 `diffusion_model_type` 是：

```text
x_start
```

此时：

```text
target = x0.detach()
x0_hat = pred
```

如果使用 `score` 模式，`target` 和 `x0_hat` 的恢复公式会不同，已经在 `train_epoch.py::_recover_x0_from_prediction` 中处理。

相关文件：

| 文件 | 作用 |
|---|---|
| `diffusion_planner/model/module/decoder.py` | DiT decoder 和 inference sampling |
| `diffusion_planner/model/diffusion_utils/sampling.py` | DPM-Solver sampler |
| `diffusion_planner/model/diffusion_utils/sde.py` | VPSDE 定义 |
| `diffusion_planner/train_epoch.py` | 训练时加噪、恢复 `x0_hat`、计算 loss |

### 3.5 Hybrid 输出：continuous head 用于 sim

历史上 hard token decode 出现过严重问题：

```text
DiT -> token IDs -> TokenTrajectoryDecoder -> trajectory
```

问题包括：

```text
静止 token 过多
token 拼接不连续
端点横向偏移过大
LQR tracker SVD did not converge
closed-loop sim 分数很低
```

因此当前主线方案改为 hybrid continuous：

```text
DiT -> x0_hat -> continuous_head -> [B, P, 80, 4]
```

`continuous_head` 输出：

```text
[x, y, cos_h, sin_h]
```

然后 planner 再转换成：

```text
[x, y, heading]
```

传给 nuPlan 的 `transform_predictions_to_states`。

当前 `continuous_head` 结构：

```text
LayerNorm(16 * D)
Linear(16 * D -> hidden)
GELU
Linear(hidden -> 80 * 4)
```

默认：

```text
D = 64
hidden = 512
output = 80 * 4 = 320
```

相关文件：

| 文件 | 作用 |
|---|---|
| `diffusion_planner/model/module/decoder.py` | `continuous_head` 和 `continuous_from_latent` |
| `diffusion_planner/train_epoch.py` | continuous trajectory loss |
| `sim_diffusion_planner_runner.sh` | 强制 sim 走 `continuous_head` |

## 4. 当前模型结构

### 4.1 顶层模型

顶层模型是：

```text
Diffusion_Planner
    encoder: Diffusion_Planner_Encoder
    decoder: Diffusion_Planner_Decoder
```

相关文件：

```text
diffusion_planner/model/diffusion_planner.py
```

### 4.2 Encoder

Encoder 负责把场景条件编码成 context tokens。

输入包括：

```text
neighbor_agents_past
static_objects
lanes
lanes_speed_limit
lanes_has_speed_limit
```

Encoder 子模块：

| 模块 | 输入 | 作用 |
|---|---|---|
| `AgentFusionEncoder` | neighbor history | 编码动态交通参与者 |
| `StaticFusionEncoder` | static objects | 编码静态障碍物 |
| `LaneFusionEncoder` | lane vectors | 编码车道、限速、交通灯 |
| `FusionEncoder` | all context tokens | 融合 agent/static/lane token |

需要注意：

```text
route_lanes 不在主 Encoder 里融合。
route_lanes 在 Decoder 的 RouteEncoder 中作为 DiT 条件使用。
```

相关文件：

```text
diffusion_planner/model/module/encoder.py
```

### 4.3 Decoder

Decoder 负责：

```text
1. 构建 token embedding table
2. 构建 token classifier head
3. 构建 continuous head
4. 调用 DiT 进行 diffusion denoising
5. 推理时根据模式输出 trajectory
```

推理路径由参数控制：

| 参数 | 作用 |
|---|---|
| `trajectory_output_mode=continuous_head` | 使用 continuous head 输出轨迹 |
| `trajectory_output_mode=token` | 使用 token decode 输出轨迹 |
| `token_selection_mode=classifier` | token 分支用 classifier 选 token |
| `token_selection_mode=nearest` | token 分支用 nearest embedding 选 token |
| `token_decode_mode=hard` | token 分支硬解码 |
| `token_decode_mode=soft` | token 分支可微/软解码实验路径 |

当前推荐 sim 模式：

```text
use_continuous_head = True
trajectory_output_mode = continuous_head
```

相关文件：

```text
diffusion_planner/model/module/decoder.py
```

### 4.4 DiT

DiT 是 diffusion denoising network。

输入：

```text
x: [B, P, 16 * D]
t: [B]
cross_c: scene context from Encoder
route_lanes: route condition
neighbor_current_mask: invalid neighbor mask
```

输出：

```text
[B, P, 16 * D]
```

其中：

| 符号 | 含义 |
|---|---|
| `B` | batch size |
| `P` | 1 + predicted_neighbor_num |
| `D` | token embedding dimension |
| `16` | 8 秒未来轨迹的 16 个 motion token |

## 5. 当前训练 loss 构成

当前 hybrid 总 loss 定义在：

```text
diffusion_planner/train_epoch.py
```

公式：

```text
loss =
    alpha_planning_loss * loss_ego
    + loss_nbr
    + lambda_token_cls_ce * loss_token_cls
    + lambda_continuous_traj * loss_continuous
    + lambda_emb_commit * loss_emb_commit
```

当前 `torch_run.sh` 配置：

```text
alpha_planning_loss = 1.0
lambda_token_cls_ce = 0.3
lambda_continuous_traj = 1.0
lambda_emb_commit = 0.0
learnable_token_emb = False
```

因此当前实际 loss 是：

```text
loss =
    1.0 * ego latent MSE
    + neighbor latent MSE
    + 0.3 * token CE
    + 1.0 * continuous trajectory SmoothL1
```

### 5.1 `loss_ego`

`loss_ego` 是 ego latent MSE。

它约束：

```text
pred_ego / x0_hat_ego 接近 GT token embedding latent
```

当前还使用了 `scene_w` 场景权重：

```text
快速/位移大的 ego 轨迹权重更高
```

作用是缓解模型倾向于预测低速或停滞轨迹的问题。

### 5.2 `loss_nbr`

`loss_nbr` 是 neighbor latent MSE。

它只对有效 neighbor 计算，使用 `nbr_valid_mask` 屏蔽无效 agent。

### 5.3 `loss_token_cls`

`loss_token_cls` 是 token classifier cross entropy（交叉熵）。

公式：

```text
loss_token_cls = loss_token_cls_ego + 0.5 * loss_token_cls_nbr
```

它约束：

```text
x0_hat token latent -> classifier -> vocab logits -> GT token IDs
```

注意：

```text
token classifier 当前是辅助监督。
sim 主路径不使用 classifier 输出的 token IDs。
```

### 5.4 `loss_continuous`

`loss_continuous` 是 continuous head 的 SmoothL1 loss。

公式：

```text
loss_continuous = loss_cont_ego + 0.5 * loss_cont_nbr
```

它约束：

```text
x0_hat -> continuous_head -> [x, y, cos_h, sin_h]
```

与 GT future trajectory 的：

```text
[x, y, cos(heading), sin(heading)]
```

做 SmoothL1。

这是当前和 sim 最相关的 loss。

### 5.5 `loss_emb_commit`

`loss_emb_commit` 只在以下条件同时满足时生效：

```text
learnable_token_emb = True
lambda_emb_commit > 0
```

当前配置中：

```text
learnable_token_emb = False
lambda_emb_commit = 0.0
```

所以它不参与训练。

## 6. TensorBoard 重点指标

建议重点看：

| 指标 | 含义 | 优先级 |
|---|---|---|
| `train_loss/loss` | 总 loss | 中 |
| `train_loss/ego_planning_loss` | ego latent MSE | 高 |
| `train_loss/neighbor_prediction_loss` | neighbor latent MSE | 中 |
| `train_loss/ego_continuous_traj_loss` | ego continuous head loss | 最高 |
| `train_loss/continuous_traj_loss` | ego+nbr continuous loss | 最高 |
| `train_loss/ego_token_cls_ce_loss` | ego token CE | 中 |
| `train_loss/ego_token_cls_acc` | ego token classifier accuracy | 中 |
| `train_loss/ego_cls_total_dx` | classifier 预测 token 总进度 | 诊断用 |
| `train_loss/ego_gt_total_dx` | GT token 总进度 | 诊断用 |
| `train_loss/ego_cls_slow_ratio` | classifier 慢 token 比例 | 诊断用 |

判断原则：

```text
如果 continuous loss 很好但 sim 很差，优先怀疑 closed-loop 分布偏移、控制器适配、轨迹动态约束不足。
如果 continuous loss 也不好，优先继续训练或调 continuous head/loss。
如果 token CE 很好但 sim 差，不代表失败，因为 sim 不走 token 输出。
```

## 7. 训练流程

### 7.1 数据准备

首先从 nuPlan 数据生成 `.npz`：

```bash
bash data_process.sh
```

或直接运行：

```bash
python data_process.py
```

生成后的 `.npz` 应包含：

```text
ego_current_state
ego_agent_future
neighbor_agents_past
neighbor_agents_future
lanes
route_lanes
static_objects
...
```

### 7.2 构建词表

使用：

```bash
python vocab_divide_token_v1.py
```

产物通常包括：

```text
ego_vocab_*.npz
nbr_vocab_*.npz
```

当前推荐词表是 0511 rolling-reference vocab：

```text
ego_vocab_0511_1024.npz
nbr_vocab_0511_1024.npz
```

### 7.3 Tokenize 数据

使用：

```bash
python tokenize_npz.py \
  --vocab /path/to/ego_vocab.npz \
  --nbr_vocab /path/to/nbr_vocab.npz \
  --data_dir /path/to/npz_dataset \
  --workers 48
```

完成后每个 `.npz` 应包含：

```text
ego_token_ids
neighbor_token_ids
```

### 7.4 清理旧增强字段

如果历史 `.npz` 里有旧的 `_aug` 字段或污染字段，可使用：

```bash
python cleanup_aug_fields.py
```

目的：

```text
避免 dataset 误读旧字段，保证训练使用当前 tokenization 结果。
```

### 7.5 启动训练

当前主训练脚本：

```bash
bash torch_run.sh
```

关键配置：

```text
EXP_NAME=training_hybrid_continuous_token_cls_ce03
batch_size=576
num_workers=16
train_epochs=1000
use_data_augment=True
alpha_planning_loss=1.0
lambda_continuous_traj=1.0
lambda_token_cls_ce=0.3
learnable_token_emb=False
use_continuous_head=True
trajectory_output_mode=continuous_head
use_token_classifier=True
token_selection_mode=classifier
```

训练输出：

```text
yzy_output/training_log/{EXP_NAME}/{timestamp}/
    args.json
    latest.pth
    model_epoch_*.pth
```

`args.json` 很重要，sim 时会读取它恢复训练配置。

## 8. 仿真流程

当前 sim 脚本：

```bash
bash sim_diffusion_planner_runner.sh
```

关键设置：

```text
SPLIT=val14
CHALLENGE=closed_loop_nonreactive_agents
USE_CONTINUOUS_HEAD=true
TRAJECTORY_OUTPUT_MODE=continuous_head
```

重要原则：

```text
当前推荐 sim 必须走 continuous_head。
不要让 hybrid checkpoint 在 sim 中退回 token hard decode。
```

脚本会传入 Hydra override：

```text
++planner.diffusion_planner.config.use_continuous_head=true
++planner.diffusion_planner.config.trajectory_output_mode=continuous_head
```

### 8.1 sim 时实际模型加载

`planner.py` 初始化时会打印：

```text
[DiffusionPlanner] ckpt=..., epoch=..., ema=True,
trajectory_output_mode=continuous_head,
use_continuous_head=True,
token_selection_mode=...,
use_token_classifier=...
```

这行用于确认：

```text
1. ckpt 路径是否正确
2. 是否加载 EMA
3. 是否真的走 continuous_head
4. 是否误用了 token mode
```

### 8.2 sim 输出转换

模型输出：

```text
[x, y, cos_h, sin_h]
```

planner 转换：

```text
heading = atan2(sin_h, cos_h)
[x, y, heading]
```

然后调用：

```text
transform_predictions_to_states
```

生成 nuPlan 的 `InterpolatedTrajectory`。

### 8.3 debug 文件

sim 会写：

```text
trajectory_debug_val14.csv
ego_token_debug_val14.csv
```

continuous_head 模式下，`ego_token_debug` 可能没有意义，因为最终不走 token 输出。

重点看 `trajectory_debug`：

| 字段 | 含义 |
|---|---|
| `finite_raw` | 原始输出是否全 finite |
| `finite_xyh` | 转换后的 xyh 是否全 finite |
| `end_x` | 8s 终点 x |
| `end_y` | 8s 终点 y |
| `step_min` | 单步最小位移 |
| `step_max` | 单步最大位移 |
| `step_mean` | 单步平均位移 |
| `zero_step_ratio` | 接近静止的帧比例 |
| `heading_jump_max` | 最大 heading 跳变 |
| `cossin_norm_min/max` | cos/sin 方向向量范数 |

判断：

```text
end_x 太小 -> 车不走，Progress/Making 差。
end_y 太大 -> 横向跑偏，Drivable/Direction 差。
step_max 太大 -> 轨迹跳变，Comfort/LQR/SVD 风险。
zero_step_ratio 太高 -> 停滞，Progress 差。
heading_jump_max 太大 -> 控制器不稳定。
```

## 9. 评估指标理解

nuPlan score 不是普通平均分，而是乘法门控结构。

常见关键指标：

| 指标 | 中文解释 | 影响 |
|---|---|---|
| `Score` | 综合分 | 总结果 |
| `Collisions` | 碰撞 | 乘法项，低了会严重拉低总分 |
| `TTC` | Time-to-collision，碰撞时间安全性 | 安全性 |
| `Drivable` | 是否在可行驶区域 | 乘法项 |
| `Comfort` | 舒适性，速度/加速度/jerk/yaw rate 等 | 常见归零项 |
| `Progress` | 沿路线前进程度 | 低速模型常低 |
| `Direction` | 行驶方向是否正确 | 路线方向相关 |
| `Making` | 是否达到推进目标 | 乘法项 |
| `SpeedLimit` | 是否遵守限速 | 次要但会影响均值 |

经验判断：

```text
Score=0 不一定表示所有指标都差。
只要乘法门控项里有关键项为 0，总分就可能是 0。
```

## 10. 历史问题总结

### 10.1 hard token decode 导致 SVD

之前尝试：

```text
DiT -> classifier/nearest -> token IDs -> hard decode -> sim
```

遇到：

```text
numpy.linalg.LinAlgError: SVD did not converge
ValueError: On entry to DLASCL parameter number ... had an illegal value
angle is not finite
```

根因不是单纯数值 NaN，而是 token 拼接后的轨迹对 nuPlan LQR tracker 不友好：

```text
相邻 token 不连续
局部 heading 和 xy derivative 不一致
大量静止 token
横向偏移过大
轨迹跳变导致曲率拟合病态
```

尝试过的修复：

```text
clip step
smooth xy
recompute heading
fallback path
transition rerank
soft decode
```

结论：

```text
这些只能缓解 SVD，不能真正提升 closed-loop 规划质量。
```

### 10.2 token classifier accuracy 看起来好，但 sim 差

原因：

```text
token 分类正确率是 open-loop token-level 指标。
closed-loop sim 需要轨迹动态、地图、路线、控制器都稳定。
token accuracy 上升不等于 sim score 上升。
```

特别是 hard token decode 下：

```text
一个 token 单独合理，不代表 16 个 token 拼起来整体合理。
```

### 10.3 训练 loss 好，但 sim 差

这是当前最核心的问题。

可能原因：

```text
1. 训练是 open-loop teacher-forced，sim 是 closed-loop rollout。
2. loss 主要看每帧误差，不直接优化 nuPlan 乘法指标。
3. continuous_head 输出未显式加入动态可行性约束。
4. route/drivable/compliance 没有作为强 loss 进入训练。
5. diffusion sampling 有随机性，闭环会放大小误差。
6. LQR tracker 对轨迹曲率、heading、速度连续性非常敏感。
```

### 10.4 词表质量影响 latent 学习

分析过的现象：

```text
新 ego vocab 中存在若干高频近静止 token。
new nbr vocab 的静止/倒退/横向 token 比例偏高。
```

风险：

```text
如果 GT tokenization 大量选择慢 token，模型会学习到保守/停滞 prior。
neighbor token 质量差会污染 shared latent training。
```

当前 hybrid 减少了 hard token decode 的风险，但 token latent MSE 和 token CE 仍然会受 vocab 分布影响。

## 11. 当前关键文件说明

| 路径 | 作用 |
|---|---|
| `train_predictor.py` | 训练入口，解析参数、创建模型、DataLoader、EMA、scheduler |
| `torch_run.sh` | 当前主训练脚本 |
| `sim_diffusion_planner_runner.sh` | 当前主 sim 脚本 |
| `data_process.py` | nuPlan scenario -> `.npz` 数据 |
| `data_process.sh` | 数据预处理启动脚本 |
| `tokenize_npz.py` | 给 `.npz` 添加 token IDs |
| `cleanup_aug_fields.py` | 清理旧增强字段 |
| `read_eval_results.py` | 读取 nuPlan sim 输出结果 |
| `diffusion_planner/model/diffusion_planner.py` | 顶层模型 |
| `diffusion_planner/model/module/encoder.py` | 场景 Encoder |
| `diffusion_planner/model/module/decoder.py` | token embedding、classifier、continuous head、DiT |
| `diffusion_planner/model/module/dit.py` | DiT block |
| `diffusion_planner/train_epoch.py` | 当前 hybrid 训练 loop |
| `diffusion_planner/utils/dataset.py` | 读取 `.npz` 和 token IDs |
| `diffusion_planner/utils/token_trajectory_decoder.py` | token IDs -> continuous trajectory |
| `diffusion_planner/utils/diff_decode.py` | soft/differentiable decode 实验工具 |
| `diffusion_planner/planner/planner.py` | nuPlan sim planner 接口 |
| `diffusion_planner/utils/config.py` | 从 args.json 和 Hydra overrides 构建 config |
| `diffusion_planner/utils/normalizer.py` | 输入 normalizer |
| `method_token_direct/` | 直接 token 分类方法实验分支 |
| `method_token_hybrid_continuous/` | hybrid continuous 方法说明和启动脚本 |

## 12. 当前推荐训练配置

推荐从以下配置开始：

```text
use_continuous_head=True
trajectory_output_mode=continuous_head
continuous_head_hidden_dim=512
lambda_continuous_traj=1.0
alpha_planning_loss=1.0
lambda_token_cls_ce=0.3
learnable_token_emb=False
lambda_emb_commit=0.0
use_token_classifier=True
token_selection_mode=classifier
use_data_augment=True
```

不建议当前立即使用：

```text
trajectory_output_mode=token
token_decode_mode=hard
learnable_token_emb=True without lambda_emb_commit
lambda_token_cls_ce very large
```

## 13. 当前推荐 sim 配置

推荐：

```text
USE_CONTINUOUS_HEAD=true
TRAJECTORY_OUTPUT_MODE=continuous_head
USE_TOKEN_CLASSIFIER=auto
TOKEN_SELECTION_MODE=auto
TOKEN_DECODE_MODE=auto
```

含义：

```text
强制最终轨迹走 continuous_head。
token 相关选项不再影响最终轨迹输出。
```

如果复现实验旧 hard-token 模型，可以设置：

```text
USE_CONTINUOUS_HEAD=auto
TRAJECTORY_OUTPUT_MODE=auto
```

但这不是当前推荐路径。

## 14. 验证流程建议

每次训练一个新模型后，不建议直接跑大规模 sim。建议按以下顺序验证。

### 14.1 验证 args.json

检查：

```text
use_continuous_head
trajectory_output_mode
lambda_continuous_traj
lambda_token_cls_ce
learnable_token_emb
vocab_path
nbr_vocab_path
```

确保训练配置和预期一致。

### 14.2 验证 sim 启动日志

查看：

```text
[DiffusionPlanner] ...
```

确认：

```text
trajectory_output_mode=continuous_head
use_continuous_head=True
ckpt path 正确
epoch 正确
ema=True
```

### 14.3 小规模 sim

先跑：

```text
val14
closed_loop_nonreactive_agents
少量 scenario
```

查看：

```text
trajectory_debug_val14.csv
runner_report.parquet
metric summary
```

### 14.4 轨迹 debug 判断

优先检查：

```text
end_x
end_y
step_max
step_mean
zero_step_ratio
heading_jump_max
```

不要只看 TensorBoard training loss。

### 14.5 再跑完整 sim

只有在小规模 sim 中轨迹数值合理后，再跑完整 `val14` 或挑战集。

## 15. 已知风险和下一步方向

### 15.1 continuous head 缺少动态约束

当前 continuous head 直接回归 `[x, y, cos, sin]`。

它不显式约束：

```text
速度连续性
加速度连续性
jerk
曲率
heading 和 xy derivative 一致性
车辆动力学可跟踪性
```

这会直接影响：

```text
Comfort
LQR stability
SVD failures
```

后续建议加入：

```text
velocity smoothness loss
acceleration smoothness loss
heading-from-xy consistency loss
curvature regularization
```

但需要谨慎加权，不要让模型只学平滑而不前进。

### 15.2 缺少 map-aware loss

当前训练 loss 不直接优化：

```text
drivable area
route progress
direction compliance
speed limit
collision
```

这解释了为什么 open-loop loss 好但 sim 分数低。

后续可以考虑：

```text
route-aligned progress loss
lane centerline distance loss
drivable boundary penalty
collision-aware loss
```

### 15.3 diffusion sampling 随机性

推理时：

```text
xT = torch.randn(...)
DPM-Solver denoise
```

每一步 closed-loop 都可能有采样随机性。

后续可以尝试：

```text
fixed noise seed for eval
multi-sample candidate reranking
temperature / noise scale control
deterministic latent initialization
```

### 15.4 neighbor loss 可能拖累 ego

neighbor prediction 对 sim 最终 ego 控制不是直接输出，但它参与训练并共享模型容量。

如果 neighbor vocab 质量差，可能影响 latent space。

后续可以尝试：

```text
降低 neighbor continuous loss 权重
降低 neighbor token CE 权重
只强化 ego continuous head
分离 ego/nbr head
```

### 15.5 vocab 分布仍需继续检查

需要长期监控：

```text
slow token ratio
negative dx token ratio
high lateral token ratio
token usage distribution
train/sim selected token distribution
```

尤其是 ego token 的总进度分布和 GT future 的总进度是否一致。

## 16. 常见问题

### 16.1 为什么 token CE 降了，sim 仍然差？

因为 sim 不直接使用 token 分类结果。CE 只说明 latent 能被分类成 token，不说明 continuous trajectory 可驾驶。

### 16.2 为什么 continuous loss 很低，sim 仍然差？

可能是 open-loop 误差低，但 closed-loop 累积误差大；也可能是轨迹帧级误差低，但曲率/舒适性/路线约束差。

### 16.3 为什么 SVD 报错？

nuPlan LQR tracker 会对轨迹拟合曲率。如果轨迹存在跳变、重复点、heading 不连续、NaN/Inf 或曲率病态，就可能出现 SVD 不收敛。

### 16.4 fallback 能解决吗？

fallback 只能避免崩溃，不能提升真实性能。过强 fallback 会让轨迹脱离模型意图，也可能导致 Drivable/Comfort/Progress 变差。

### 16.5 当前最应该优化什么？

优先优化 sim 输出轨迹本身：

```text
continuous_head 轨迹质量
动态连续性
route progress
closed-loop 稳定性
```

而不是继续单独追求 token classification accuracy。

## 17. 当前仓库推送状态

当前代码快照已推送到：

```text
https://github.com/Jasonyu728/hybrid-dp-planner
```

推送分支：

```text
main
```

当前仓库 `.gitignore` 已排除：

```text
npz_dataset/
npz2token_dataset/
yzy_output/
backup/
work_md/
vocab_old/
*.pth
*.ckpt
*.parquet
*.tfevents*
.claude/settings.local.json
```

因此仓库只应保存代码、配置、小型词表和说明文档，不保存大型训练数据、checkpoint 或仿真输出。

## 18. 一句话总结

当前项目的核心不是“把轨迹离散化后直接输出 token”，而是：

```text
用 motion token 构建更结构化的 diffusion latent space，
用 token CE 辅助约束 latent，
最终用 continuous_head 输出可用于 nuPlan sim 的连续轨迹。
```

下一阶段的重点应从“训练 loss 是否下降”转向“closed-loop sim 轨迹是否可驾驶、可跟踪、符合路线和舒适性约束”。

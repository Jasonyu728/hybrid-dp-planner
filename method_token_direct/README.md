# method_token_direct

这一版用于验证“直接预测 token 编号”是否能改善仿真时选到低速 token 的问题。

## 核心变化

- 训练时直接使用 token 分类损失。
- 仿真时需要使用 `token_selection_mode=classifier`，也就是用分类头直接选 token。
- 固定 token embedding，不再让 embedding 参与学习。
- 不使用 reconstruction loss、codebook-distance CE、progress loss、low-speed margin loss。

## 当前损失

```text
loss =
  alpha_planning_loss * ego_diffusion_loss
  + neighbor_diffusion_loss
  + lambda_token_cls_ce * token_classifier_ce_loss
```

推荐初始参数：

```text
alpha_planning_loss = 1.0
lambda_token_cls_ce = 1.0
learnable_token_emb = False
use_token_classifier = True
token_selection_mode = classifier
```

## 运行

在 Diffusion-Planner 根目录运行：

```bash
bash method_token_direct/torch_run.sh
```

注意：主工程的 `diffusion_planner/model/module/decoder.py` 必须已经包含 token classifier head，并且 sim 阶段也要使用 `token_selection_mode=classifier`。

## 重点观察指标

- `train_loss/loss`
- `train_loss/ego_planning_loss`
- `train_loss/neighbor_prediction_loss`
- `train_loss/ego_token_cls_ce_loss`
- `train_loss/ego_token_cls_acc`
- `train_loss/ego_cls_total_dx`
- `train_loss/ego_gt_total_dx`
- `train_loss/ego_cls_slow_ratio`

如果 `ego_token_cls_acc` 上升，但 `ego_cls_total_dx` 仍然很低，说明分类头仍然偏向低速 token，下一步应优先检查词表分布或改成 SMART-style token。

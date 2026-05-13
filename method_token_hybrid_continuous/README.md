# method_token_hybrid_continuous

这一版的目标是避免把 hard token 拼接轨迹直接送进 nuPlan 控制器。

核心逻辑：

- diffusion 主干仍然学习 token embedding latent，保留 token 先验。
- token classifier 只作为辅助监督，用来让 latent 更接近正确 token 编号。
- 新增 continuous trajectory head，把 latent 直接回归成连续轨迹。
- sim 阶段使用连续轨迹输出，不再用 hard token decode 作为最终控制轨迹。

推荐先跑这一版，不再启用 progress loss、low-speed margin loss、soft token decode。

训练完成后，`args.json` 里应包含：

```json
"use_continuous_head": true,
"trajectory_output_mode": "continuous_head",
"use_token_classifier": true,
"token_selection_mode": "classifier"
```

sim 时不要把这些字段覆盖成旧模式。

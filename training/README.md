# 训练与导出

RL 策略训练 → 选优 → 导出（onnx/pt/samples.json）→ 一致性验证。

## 脚本

- `checkpoint.py` — 从训练日志选奖励最高的 checkpoint
- `export.py` — 一键完成选优 + 导出 policy.onnx/policy.pt + 录制 samples.json
- `samples.py` — 记录推理样本（export.py 会调用，也可独立跑）
- `sim2sim.py` — ONNX vs PyTorch 一致性验证（验收最大误差 < 1e-4）

## 前置

Windows `isaac` 环境；`ISAACLAB_ROOT` 指向 IsaacLab 源码目录。

## 导出

```bash
ISAACLAB_ROOT=<IsaacLab 源码目录> python training/export.py \
    --log <训练 stdout 日志> --run_dir <训练日志目录>
```

输出到 `assets/models/`（policy.onnx / policy.pt / best_model.pt）和 `data/samples.json`。

## 验证

```bash
python training/sim2sim.py
```

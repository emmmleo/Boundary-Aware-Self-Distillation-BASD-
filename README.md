# BASD 主实验代码说明（Boundary-Aware Self-Distillation）

> 这个仓库当前同时包含两部分：
> 1) 早期/预实验脚本（`basd_pilot/`, `basd_protocol/`）
> 2) **主实验新实现**（`basd/`, `scripts/train_basd.py`）
>
> 本 README 重点介绍你现在要跑的 **主实验新实现**，不依赖旧的 pre-experiment 训练逻辑。

---

## 1. 方法概述

主实验采用“on-policy 自蒸馏 + 错误样本边界增强”的训练流程：

1. student（base + LoRA）先对题目做 on-policy rollout。
2. teacher 使用同一个 base model，但禁用 LoRA，并额外看到 `reference_solution` 作为 privileged context。
3. 在同一条 student completion 上，分别计算 teacher/student 的 token-level 打分。
4. 若样本答对：全序列（reasoning + final）统一权重蒸馏。
5. 若样本答错：在 reasoning token 上检测 boundary（信号跃升点），并把 boundary 映射到 step，对邻域 step 做权重提升。
6. loss 采用 OPSD 风格目标：`JSD + reverse-KL + PG`（可分别开关），词表空间可切 `full` 或 `teacher_topk`。

核心思想：
- **detect with gap / train with KL**
- **uniform base + local boost（错误样本不清零非边界 token）**

---

## 2. 项目结构（主实验相关）

```text
basd/
  types.py                      # 训练核心 dataclass
  data/
    dataset.py                  # JsonlMathDataset
    prompt_builder.py           # student/teacher prompt
    answer_extractor.py         # 抽取 final answer + 对错判断
    collator.py
  model/
    loader.py                   # tokenizer/base+LoRA 加载、teacher/student adapter 切换
    scoring.py                  # completion logits 对齐 + sampled token logprob
    peft_utils.py
  rollout/
    generator.py                # on-policy generation
    parser.py                   # <<STEP_i>> / <<FINAL>> 解析
    aligner.py                  # text span -> token 对齐
  signal/
    token_metrics.py            # sampled gap / token KL
    boundary_detector.py        # EMA + jump + persistence 检测
    weighting.py                # boundary step 邻域权重
  loss/
    distill.py                  # full KL / teacher_topk KL
    masks.py
  trainer/
    step_fn.py                  # 单步训练流程
    engine.py                   # manual loop + accelerate
    logger.py                   # JSONL debug 日志
  eval/
    metrics.py
    evaluator.py
  utils/
    config.py / seed.py / io.py / distributed.py

scripts/
  train_basd.py                 # 主训练入口
  eval_basd.py                  # 简单评估入口
  tune_boundary_from_protocol.py# 用 protocol 记录估阈值
  inspect_rollout_debug.py      # 查看 debug 样本

configs/
  basd_qwen3_8b.yaml            # 主配置
  distill_full_vocab.yaml       # distill ablation
  distill_teacher_topk.yaml     # distill ablation
  boundary_default.yaml         # boundary 默认参数
```

---

## 3. 环境安装

建议 Python 3.10+。

```bash
pip install -U torch transformers peft accelerate datasets pyyaml numpy pytest
```

> 如果你使用 Qwen3-8B，请确认 GPU 显存足够，并根据机器情况配置 bf16 / grad checkpoint / LoRA 参数。

---

## 4. 数据格式

训练数据建议是 JSONL，每行最少包含：

```json
{
  "sample_id": "gsm8k_001",
  "question": "...",
  "gold_answer": "...",
  "reference_solution": "..."
}
```

兼容字段：
- `sample_id` 缺失时可回退 `id`
- `gold_answer` 缺失时可回退 `answer`
- `reference_solution` 缺失时可回退 `solution`

---

## 5. 快速开始

### 5.1 修改配置

先检查 `configs/basd_qwen3_8b.yaml`：

- `model.base_model_name_or_path`
- `data.train_file`
- `train.num_train_steps`
- `distill.vocab_mode`（`full` 或 `teacher_topk`）

### 5.2 启动训练

```bash
python scripts/train_basd.py --config configs/basd_qwen3_8b.yaml
```

训练日志会写到：
- `output/train_runs/.../train_debug.jsonl`

最终 checkpoint 默认保存到：
- `output/train_runs/.../final_ckpt`

---

## 6. 关键配置说明

### 6.1 蒸馏模式

```yaml
distill:
  vocab_mode: teacher_topk   # full | teacher_topk
  objective: opsd_jsd_reverse_kl_pg
  w_jsd: 1.0
  w_reverse_kl: 1.0
  w_pg: 1.0
  topk: 64
```

- `full`: 在全词表上计算 JSD/reverse-KL/PG 所需概率
- `teacher_topk`: 在 teacher top-k 子词表上重归一化后计算对应项（更省算力）

### 6.2 boundary 检测

```yaml
boundary:
  signal_type: sampled_gap   # sampled_gap | token_kl
  smooth_alpha: 0.3
  abs_threshold: 0.8
  jump_threshold: 0.5
  persist_window: 8
```

- 检测建议先用 `sampled_gap`
- 训练 loss 继续用 KL

### 6.3 权重策略

```yaml
weighting:
  correct_uniform_weight: 1.0
  incorrect_fallback_uniform_weight: 1.0
  step_weight_table:
    "-2": 1.2
    "-1": 1.5
    "0": 3.0
    "1": 2.5
    "2": 1.8
```

- 正确样本：统一权重
- 错误样本：统一基础权重 + boundary 邻域 boost
- 未检测到 boundary：回退为统一权重，不丢样本

---

## 7. 评估与辅助脚本

### 7.1 评估

```bash
python scripts/eval_basd.py --records <your_records.jsonl>
```

### 7.2 基于 protocol 自动估阈值

```bash
python scripts/tune_boundary_from_protocol.py \
  --protocol_records outputs/basd_protocol_run100/records.jsonl
```

### 7.3 debug 样本查看

```bash
python scripts/inspect_rollout_debug.py \
  --debug_jsonl output/train_runs/basd_qwen3_8b/train_debug.jsonl
```

---

## 8. 测试

```bash
python -m pytest -q
```

当前包含：
- parser / boundary 检测基础单测
- distill loss 两种模式数值稳定性单测

如环境缺 `torch` 会在收集阶段报错，请先安装依赖后再跑。

---

## 9. 常见问题

### Q1: teacher 和 student 是否复制了两份 8B 权重？
没有。当前设计是同一 base model，student 打开 LoRA adapter，teacher 关闭 LoRA adapter。

### Q2: 为什么 boundary 只在 reasoning 区域检测？
为了避免“最终答案 token gap 最大导致边界总落结尾”的假阳性。

### Q3: 错误样本只训练 boundary 附近吗？
不是。采用“全局保底 + 局部增强”，更稳定。

### Q4: 如何做 OPSD 对齐对比？
把 `distill.vocab_mode` 切到 `full` 跑 baseline。

---

## 10. 与旧代码关系

- `basd_pilot/`、`basd_protocol/`：历史预实验与协议分析代码。
- `basd/` + `scripts/train_basd.py`：主实验训练框架。

建议主实验论文与复现实验优先使用 `basd/` 这套模块化实现。

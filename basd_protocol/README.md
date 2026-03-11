# BASD Protocol v2

这套实现独立于旧 `basd_pilot`，目标是严格贴合预实验 protocol：

- `run_protocol.py`
  - 生成 student rollout
  - 在同一条 rollout 上计算 teacher/student token logprob
  - 记录 `gap = teacher_logprob - student_logprob`
  - 可选记录 `token_kl`
  - 聚合 `step_metrics`
  - 产出 `records.jsonl`、`annotation_template.csv`、`summary.json`
- `analyze_protocol.py`
  - 合并人工标注
  - 导出 `sample_summary.csv`
  - 计算 exact/near match、MAE、correct vs wrong 统计
  - 输出 token-level、step-level、aggregate、heatmap 图

## 目录建议

```text
outputs/
  basd_protocol_run/
    records.jsonl
    records.partial.jsonl
    annotation_template.csv
    summary.json
  basd_protocol_analysis/
    summary.json
    sample_summary.csv
    report.md
    token_level_cases/
    step_level_cases/
    aggregate_plots/
    heatmaps/
```

## 输入 schema

每条 jsonl 至少包含：

```json
{
  "sample_id": "gsm8k-test-0",
  "question": "...",
  "reference_solution": "...",
  "gold_answer": "18"
}
```

兼容字段：

- `id` 可替代 `sample_id`
- `final_answer` 可替代 `gold_answer`
- 若没有 `gold_answer`，会从 `reference_solution` 中自动抽取

## 输出 schema

`records.jsonl` 每条记录至少包含：

```json
{
  "sample_id": "...",
  "question": "...",
  "reference_solution": "...",
  "gold_answer": "...",
  "student_full_text": "...",
  "student_final_answer": "...",
  "is_correct": false,
  "format_fail": false,
  "num_steps": 4,
  "num_tokens": 87,
  "predicted_boundary_by_gap": 3,
  "predicted_boundary_by_gap_jump": 3,
  "predicted_boundary_by_kl": 3,
  "steps": [...],
  "tokens": [...],
  "step_metrics": [...]
}
```

`annotation_template.csv` 字段：

- `sample_id`
- `is_correct`
- `first_error_step`
- `error_type`
- `boundary_pattern`
- `comments`
- `predicted_boundary_by_gap`
- `predicted_boundary_by_gap_jump`
- `predicted_boundary_by_kl`
- `num_steps`

## 运行命令

```bash
python -m basd_protocol.run_protocol \
  --input data/gsm8k_test_full.jsonl \
  --out_dir outputs/basd_protocol_run100 \
  --model_name models/Qwen3-8B \
  --temperature 0.7 \
  --top_p 0.8 \
  --top_k 20 \
  --min_p 0.0 \
  --max_new_tokens 768 \
  --max_step_count 16 \
  --compute_token_kl \
  --limit 100
```

```bash
python -m basd_protocol.analyze_protocol \
  --records outputs/basd_protocol_run/records.jsonl \
  --annotations outputs/basd_protocol_run/annotation_template.csv \
  --out_dir outputs/basd_protocol_analysis100 \
  --plot_cases 30
```

## 脚本输出

`analyze_protocol.py` 默认输出：

- 代表性 `token_level_cases/*.png`
- 代表性 `step_level_cases/*.png`
- `aggregate_plots/avg_gap_correct_vs_wrong.png`
- `aggregate_plots/avg_kl_correct_vs_wrong.png`
- `aggregate_plots/avg_gap_by_boundary_pattern.png`
- `heatmaps/step_gap_heatmap.png`
- `heatmaps/step_kl_heatmap.png`

## 说明

- `predicted_boundary_by_gap` 是 `argmax_i avg_gap_i`
- `predicted_boundary_by_gap_jump` 是 `argmax_i (avg_gap_i - avg_gap_{i-1})`
- `predicted_boundary_by_kl` 是 `argmax_i avg_kl_i`
- `format_fail=true` 的样本会保留在日志里，但不适合作为主分析样本

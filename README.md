# BASD 正式预实验（Qwen3-8B + 完整数据集）

本仓库已升级为**正式预实验版本**，目标是验证：

1. 错误样本的 step-level KL 是否呈现“前低后高 + 跃迁”；
2. 跃迁边界是否接近 first error step；
3. `max jump step` 是否优于 `max absolute KL step`。

---

## 1) 环境准备

```bash
pip install -U transformers datasets torch matplotlib accelerate
```

> 推荐使用有 GPU 的环境运行 Qwen3-8B。

---

## 1.5) 下载 Qwen3-8B 到本地（可选）

如果你希望先把模型下载到本地再运行实验，可以执行：

```bash
pip install -U huggingface_hub
python download_qwen3_8b.py --local-dir models/Qwen3-8B
```

如果使用国内镜像，可先设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python download_qwen3_8b.py --local-dir models/Qwen3-8B
```

下载后运行实验时，将 `--model_name` 改为本地目录即可（例如 `models/Qwen3-8B`）。

---

## 2) 准备完整数据集（GSM8K 全量 test）

```bash
python data/prepare_gsm8k_full.py --out data/gsm8k_test_full.jsonl
```

输出字段：
- `id`
- `question`
- `reference_solution`

---

## 3) 运行正式预实验（Qwen3-8B）

```bash
python basd_pilot/run_formal_experiment.py \
  --input data/gsm8k_test_full.jsonl \
  --out_dir outputs/formal_gsm8k_qwen3_8b \
  --model_name /scratch/azureml/yz/model/Qwen3-8B \
  --temperature 0.7 \
  --top_p 0.95 \
  --max_new_tokens 1024 \
  --limit 20

python basd_pilot/run_formal_experiment.py \
  --input data/gsm8k_test_full.jsonl \
  --out_dir outputs/formal_gsm8k_qwen3_8b_debug \
  --model_name /scratch/azureml/yz/model/Qwen3-8B \
  --temperature 0 \
  --max_new_tokens 2048 \
  --limit 20

torchrun --nproc_per_node=8 basd_pilot/run_formal_experiment.py \
  --input data/gsm8k_test_full.jsonl \
  --out_dir outputs/formal_gsm8k_qwen3_8b_fast \
  --model_name /scratch/azureml/yz/model/Qwen3-8B \
  --temperature 0.0 \
  --max_new_tokens 1024 \
  --limit 24
```

- `--limit 0` 表示跑完整数据集。
- 若仅调试，可设 `--limit 20`。

---

## 4) 方法对应关系

脚本 `basd_pilot/run_formal_experiment.py` 实现：

1. student 按 step-by-step 生成解答；
2. teacher/student 在同一条 student rollout 上逐 token 打分；
3. 聚合成 step-level KL：`D_i`；
4. 边界检测：`argmax_i (D_i - D_{i-1})`；
5. 对照基线：`argmax_i D_i`；
6. 导出人工标注模板并计算统计。

teacher/student 共享同一 backbone（Qwen3-8B），差异仅在上下文：
- teacher 看到 `question + reference_solution`
- student 仅看到 `question`

---

## 5) 输出文件

`outputs/formal_gsm8k_qwen3_8b/` 下：

- `records.jsonl`：每题完整记录（student 解答、step-KL 曲线、boundary）
- `annotation_template.csv`：人工 first-error 标注模板
- `summary.json`：全局统计

---

## 6) 人工标注与最终统计

默认运行中 `manual_first_error_step` 为 `None`，请在 `annotation_template.csv` 中补标后，回填到 `records.jsonl` 再做最终统计。

建议指标：
- exact match
- ±1 hit rate
- MAE
- max jump vs max absolute KL 对照胜率

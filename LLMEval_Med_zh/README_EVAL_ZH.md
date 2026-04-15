# LLMEval_Med 评估流程（CIA + Discrimination）

本流程在 `LLMEval_Med` 下独立运行，不会覆盖主项目已有评估结果。

## 0) 前置
- 先确保你已经生成：
  - `LLMEval_Med/outputs/rubrics_generated_zh.jsonl`
  - `LLMEval_Med/outputs/reference_rubrics_from_checklist_zh.jsonl`

## 1) 准备评估资产
```bash
python3 LLMEval_Med/eval_prepare_assets_zh.py
```

生成：
- `LLMEval_Med/outputs/eval/reference_responses_zh.jsonl`
- `LLMEval_Med/outputs/eval/gold_reference_rubrics_zh.jsonl`

## 2) CIA 评估
### 2.1 构建 Gold Atomic Points
```bash
LLM_PROVIDER=cerebras CEREBRAS_API_KEY=你的key JUDGE_MODEL=llama3.3-70b \
START_INDEX=0 END_INDEX=113 \
python3 LLMEval_Med/eval_build_gold_atomic_points_zh.py
```

输出：
- `LLMEval_Med/outputs/eval/gold_atomic_points_zh.jsonl`
- cache: `LLMEval_Med/outputs/eval/gold_atomic_cache/`

### 2.2 运行 CIA Batch
```bash
LLM_PROVIDER=cerebras CEREBRAS_API_KEY=你的key JUDGE_MODEL=llama3.3-70b \
python3 LLMEval_Med/eval_cia_batch_zh.py
```

默认模式：
- `ours`（`LLMEval_Med/outputs/rubrics_generated_zh.jsonl`）

可扩展模式（示例）：
```bash
CIA_MODES=ours,generic \
CIA_EXTRA_MODE_PATHS="generic=LLMEval_Med/outputs/eval/generic_rubrics.jsonl" \
python3 LLMEval_Med/eval_cia_batch_zh.py
```

输出：
- `LLMEval_Med/outputs/eval/cia_summary_zh.json`
- cache: `LLMEval_Med/outputs/eval/cia_cache/`

## 3) Discrimination 评估
### 3.1 生成 Perturbed Candidates
```bash
LLM_PROVIDER=cerebras CEREBRAS_API_KEY=你的key PERTURB_MODEL=llama3.3-70b \
python3 LLMEval_Med/eval_generate_perturbed_candidates_zh.py
```

输出：
- `LLMEval_Med/outputs/eval/perturbed_candidates_zh.jsonl`

### 3.2 构建 Pairwise Dataset
```bash
python3 LLMEval_Med/eval_build_pairwise_dataset_zh.py --resume
```

输出：
- `LLMEval_Med/outputs/eval/pairwise_dataset_perturbed_zh.jsonl`

### 3.3 运行 Discriminative Batch
```bash
LLM_PROVIDER=cerebras CEREBRAS_API_KEY=你的key JUDGE_MODEL=llama3.3-70b \
DISCRIM_MODES=ours,none \
python3 LLMEval_Med/eval_discriminative_batch_zh.py
```

可扩展模式（示例）：
```bash
DISCRIM_MODES=ours,none,generic \
DISCRIM_EXTRA_MODE_PATHS="generic=LLMEval_Med/outputs/eval/generic_rubrics.jsonl" \
python3 LLMEval_Med/eval_discriminative_batch_zh.py
```

输出目录：
- `LLMEval_Med/outputs/eval/discrimination/`
  - `results_{mode}.jsonl`
  - `run_summary_*.json`

### 3.4 汇总报告（含 AUROC）
```bash
python3 LLMEval_Med/eval_discrimination_report_zh.py \
  --input-dir LLMEval_Med/outputs/eval/discrimination \
  --modes ours,none,generic
```

输出：
- `LLMEval_Med/outputs/eval/discrimination/final_report_summary_zh.json`

## 4) 断点续跑
- `CIA`：删除 `cia_cache/<mode>/<index>.json` 可重跑对应 index。
- `Discrimination`：
  - 删除 `results_<mode>.jsonl` 内对应 pair 或整文件可重跑；
  - `--resume` 会自动跳过已完成记录。

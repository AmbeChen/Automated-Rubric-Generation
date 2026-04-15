# LLMEval_Med 中文 Rubrics 生成流程

## 目标
- 使用 `LLMEval_Med/problems_checklist_clean.jsonl` 跑一套与原项目结构对齐的 Rubrics 生成流程（中文版）。
- 字段映射：
  - `problem` => `query`
  - `checklist` => `reference_rubrics`（用于参考标准）

## 脚本
- `LLMEval_Med/rag_pipeline_zh.py`
  - 中文版 Stage 1 RAG（Router/No-Router、中文权威域名白名单检索、Rerank、Evidence Synthesis）
- `LLMEval_Med/rubrics_pipeline_zh.py`
  - 中文版 Rubric Pipeline（步骤结构与原流程对齐：抽取、过滤、意图、合成、精炼）
  - 支持接收 RAG 证据上下文（`evidence_data`）
- `LLMEval_Med/run_rubrics_generation_zh.py`
  - 批量运行入口
  - 支持 `--use-rag`（可选 `--no-router`）
  - 输出单条文件 + 聚合 JSONL + evidence 缓存

## 运行
在仓库根目录执行：

```bash
LLM_PROVIDER=cerebras CEREBRAS_API_KEY=你的key RUBRIC_MODEL=llama3.3-70b \
python3 LLMEval_Med/run_rubrics_generation_zh.py \
  --resume \
  --start-index 0 \
  --end-index 999999
```

启用 RAG（中文网站检索）：

```bash
LLM_PROVIDER=cerebras CEREBRAS_API_KEY=你的key RUBRIC_MODEL=llama3.3-70b \
TAVILY_API_KEY=你的key \
python3 LLMEval_Med/run_rubrics_generation_zh.py \
  --use-rag \
  --resume \
  --start-index 0 \
  --end-index 999999
```

No Router（跳过意图分析与关键词扩展，直接用原始 Query 检索）：

```bash
LLM_PROVIDER=cerebras CEREBRAS_API_KEY=你的key RUBRIC_MODEL=llama3.3-70b \
TAVILY_API_KEY=你的key \
python3 LLMEval_Med/run_rubrics_generation_zh.py \
  --use-rag \
  --no-router \
  --resume
```

可选 provider：
- `LLM_PROVIDER=openai` + `OPENAI_API_KEY`
- `LLM_PROVIDER=groq` + `GROQ_API_KEY`

## 输出目录
- `LLMEval_Med/outputs/rubrics_individual_zh/rubric_{index}.json`
  - 单条详细结果（含 `full_trace`）
- `LLMEval_Med/outputs/rubrics_generated_zh.jsonl`
  - 生成 Rubrics 聚合结果
- `LLMEval_Med/outputs/reference_rubrics_from_checklist_zh.jsonl`
  - checklist 转换后的参考 Rubrics 聚合结果
- `LLMEval_Med/outputs/evidence_zh/conversation_{index}.json`（启用 `--use-rag` 时）
  - Stage 1 检索结果缓存（支持断点续跑）

## 结果结构说明
- `generated_rubrics`:
  - `{"rubrics": [{"criterion": "...", "axis": "...", "points": ...}, ...]}`
- `reference_rubrics`:
  - 由 `checklist` 逐条映射为 rubric 项（默认 `axis=completeness`, `points=6`）

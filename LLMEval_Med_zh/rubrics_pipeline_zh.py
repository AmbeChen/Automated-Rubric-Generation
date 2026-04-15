import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(round(len(text) / 4)))


def _usage_get(usage_obj, key: str):
    if usage_obj is None:
        return None
    if isinstance(usage_obj, dict):
        return usage_obj.get(key)
    return getattr(usage_obj, key, None)


def clean_json_output(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        return {}
    t = text.strip()
    if not t:
        return {}

    t = t.replace("Infinity", "100").replace("-Infinity", "-100")
    try:
        return json.loads(t)
    except Exception:
        pass

    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    m = re.search(r"(\{[\s\S]*\})", t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return {}


def _extract_error_message(err: Exception) -> str:
    if err is None:
        return ""
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        nested = body.get("error")
        if isinstance(nested, dict):
            msg = nested.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
        msg = body.get("message")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()
    return str(err).strip()


class UniversalLLMClientZH:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "cerebras").lower()
        self.base_url, self.api_key = self._resolve_provider(self.provider)
        self.model = os.getenv("RUBRIC_MODEL", self._default_model(self.provider))
        self.timeout = float(os.getenv("RUBRIC_TIMEOUT", "60"))
        self.temperature = float(os.getenv("RUBRIC_TEMPERATURE", "0.1"))
        self.use_json_mode = os.getenv("RUBRIC_USE_JSON_MODE", "1") == "1"
        self.last_call_meta: Dict[str, Any] = {}
        self.call_history: List[Dict[str, Any]] = []

        if not self.api_key:
            raise RuntimeError(
                "Missing API key. Set OPENAI_API_KEY / CEREBRAS_API_KEY / GROQ_API_KEY."
            )
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    @staticmethod
    def _resolve_provider(provider: str) -> Tuple[str, Optional[str]]:
        if provider == "groq":
            return (
                "https://api.groq.com/openai/v1",
                os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"),
            )
        if provider == "cerebras":
            return (
                "https://api.cerebras.ai/v1",
                os.getenv("CEREBRAS_API_KEY") or os.getenv("OPENAI_API_KEY"),
            )
        return ("https://api.openai.com/v1", os.getenv("OPENAI_API_KEY"))

    @staticmethod
    def _default_model(provider: str) -> str:
        if provider == "groq":
            return "llama-3.3-70b-versatile"
        if provider == "cerebras":
            return "llama3.3-70b"
        return "gpt-4.1-mini"

    def reset_metrics(self):
        self.last_call_meta = {}
        self.call_history = []

    def create_completion(self, system_prompt: str, user_prompt: str) -> str:
        start = time.perf_counter()
        est_prompt_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }

        try:
            if self.use_json_mode:
                try:
                    resp = self.client.chat.completions.create(
                        **kwargs, response_format={"type": "json_object"}
                    )
                except Exception as e:
                    msg = _extract_error_message(e).lower()
                    if "response_format" in msg or "json_object" in msg:
                        resp = self.client.chat.completions.create(**kwargs)
                    else:
                        raise
            else:
                resp = self.client.chat.completions.create(**kwargs)

            content = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            prompt_tokens = _usage_get(usage, "prompt_tokens")
            completion_tokens = _usage_get(usage, "completion_tokens")
            total_tokens = _usage_get(usage, "total_tokens")

            if prompt_tokens is None:
                prompt_tokens = est_prompt_tokens
            if completion_tokens is None:
                completion_tokens = _estimate_tokens(content)
            if total_tokens is None:
                total_tokens = int(prompt_tokens) + int(completion_tokens)

            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            meta = {
                "provider": self.provider,
                "model": self.model,
                "latency_ms": elapsed_ms,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(total_tokens),
                "usage_source": "api" if usage is not None else "estimated",
                "error": None,
            }
            self.last_call_meta = meta
            self.call_history.append(meta)
            return content
        except Exception as e:
            err_text = _extract_error_message(e)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            meta = {
                "provider": self.provider,
                "model": self.model,
                "latency_ms": elapsed_ms,
                "prompt_tokens": int(est_prompt_tokens),
                "completion_tokens": 0,
                "total_tokens": int(est_prompt_tokens),
                "usage_source": "estimated",
                "error": err_text,
            }
            self.last_call_meta = meta
            self.call_history.append(meta)
            return json.dumps({"error": err_text}, ensure_ascii=False)


def extract_rubrics_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        rubrics = obj.get("rubrics")
        if isinstance(rubrics, list):
            return {"rubrics": _normalize_rubrics(rubrics)}
        return {"rubrics": []}
    if isinstance(obj, list):
        return {"rubrics": _normalize_rubrics(obj)}
    return {"rubrics": []}


def _normalize_rubrics(items: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    allowed_axes = {
        "accuracy",
        "completeness",
        "context_awareness",
        "communication_quality",
        "instruction_following",
    }
    for x in items:
        if not isinstance(x, dict):
            continue
        criterion = str(x.get("criterion", "")).strip()
        if not criterion:
            continue
        axis = str(x.get("axis", "completeness")).strip()
        if axis not in allowed_axes:
            axis = "completeness"
        try:
            points = int(round(float(x.get("points", 0))))
        except Exception:
            points = 0
        points = max(-10, min(10, points))
        out.append({"criterion": criterion, "axis": axis, "points": points})
    return out


class RubricPipelineZH:
    def __init__(
        self,
        query: str,
        llm_client: UniversalLLMClientZH,
        evidence_data: Optional[Dict[str, Any]] = None,
    ):
        self.query = query
        self.llm_client = llm_client
        self.evidence_data = evidence_data or {}
        self.evidence_text = self._build_evidence_text()
        self.debug_trace: Dict[str, Any] = {
            "step_0_retrieval": self.evidence_data,
            "step_1_1_atomic_facts": {},
            "step_1_2_filtered_facts": {},
            "step_2_interaction_intent": {},
            "step_3_draft_rubrics": {},
            "step_4_final_rubrics": {},
            "metrics": {
                "steps": {},
                "llm_calls": [],
            },
        }

    def _build_evidence_text(self) -> str:
        if not isinstance(self.evidence_data, dict) or not self.evidence_data:
            return ""
        parts: List[str] = []
        synthesis = self.evidence_data.get("synthesis", {})
        if isinstance(synthesis, dict):
            consensus = str(synthesis.get("consensus", "")).strip()
            if consensus:
                parts.append(f"共识: {consensus}")
            red_flags = synthesis.get("red_flags")
            if isinstance(red_flags, list) and red_flags:
                parts.append(f"风险警示: {json.dumps(red_flags, ensure_ascii=False)}")
            contention = str(synthesis.get("contention", "")).strip()
            if contention:
                parts.append(f"争议点: {contention}")

        sources = self.evidence_data.get("evidence_sources", [])
        if isinstance(sources, list):
            for i, s in enumerate(sources[:8], 1):
                if not isinstance(s, dict):
                    continue
                src = str(s.get("source_name", "Unknown")).strip()
                url = str(s.get("url", "")).strip()
                excerpt = str(s.get("key_excerpt", "")).strip()
                if excerpt and len(excerpt) > 1200:
                    excerpt = excerpt[:1200]
                if excerpt:
                    parts.append(f"[来源{i}] {src} ({url})\n{excerpt}")
                else:
                    parts.append(f"[来源{i}] {src} ({url})")
        return "\n".join(parts).strip()

    def _run_step(self, step_name: str, fn):
        calls_before = len(self.llm_client.call_history)
        t0 = time.perf_counter()
        fn()
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        calls_after = len(self.llm_client.call_history)
        new_calls = self.llm_client.call_history[calls_before:calls_after]

        prompt_tokens = sum(int(c.get("prompt_tokens", 0)) for c in new_calls)
        completion_tokens = sum(int(c.get("completion_tokens", 0)) for c in new_calls)
        total_tokens = sum(int(c.get("total_tokens", 0)) for c in new_calls)
        errors = [c.get("error") for c in new_calls if c.get("error")]

        self.debug_trace["metrics"]["steps"][step_name] = {
            "latency_ms": elapsed_ms,
            "llm_calls": len(new_calls),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "errors": errors,
        }
        for c in new_calls:
            cc = dict(c)
            cc["step"] = step_name
            self.debug_trace["metrics"]["llm_calls"].append(cc)

    def step_1_1_atomic_extraction(self):
        system_prompt = "你是医学评估标准设计专家。"
        evidence_block = self.evidence_text or "（暂无外部检索证据，仅基于问题本身抽取）"
        user_prompt = f"""
任务：根据用户问题和检索证据，提取用于后续评估回答质量的“核心医学细节块（Semantic Clinical Details）”。

用户问题：
{self.query}

检索证据（RAG）：
{evidence_block}

严格遵守以下原则：
必须原封不动地保留证据中的具体药物名称、时间界限（如“发病2小时内”）、特定生理指标和具体症状。
不要剥离因果关系和条件限制，例如“如果患者有X症状”或“在使用某药物后”。
只需剔除完全无关的口语化铺垫或冗余背景，保留所有具备临床操作指导意义的“有用片段”。

输出要求：
1) 只输出 JSON 格式。
2) JSON 结构必须是：
{{
  "essential_clinical_details": ["..."],
  "contraindications_and_errors": ["..."],
  "critical_safety_red_flags": ["..."]
}}
3) essential_clinical_details: 回答中必须覆盖的具体医学事实、需要追问的详细病史或具体处理流程。
4) contraindications_and_errors: 明确的禁忌症、不适用的药物或容易犯的具体临床逻辑错误。
5) critical_safety_red_flags: 具体的危险信号（Red flags）和触发紧急就医的确切指征。
""".strip()
        res = self.llm_client.create_completion(system_prompt, user_prompt)
        self.debug_trace["step_1_1_atomic_facts"] = clean_json_output(res)

    def step_1_2_filtering(self):
        system_prompt = "你是医学评估标准过滤器。"
        evidence_block = self.evidence_text or "（暂无外部检索证据）"
        user_prompt = f"""
任务：对原子事实进行相关性筛选，保留与该问题直接相关且可用于评分的内容。

用户问题：
{self.query}

检索证据（RAG）：
{evidence_block}

原始原子事实：
{json.dumps(self.debug_trace.get("step_1_1_atomic_facts", {}), ensure_ascii=False)}

输出 JSON：
{{
  "relevant_positive_facts": ["..."],
  "relevant_negative_constraints": ["..."],
  "relevant_red_flags": ["..."]
}}

规则：
- 只保留与当前问题直接相关的条目。
- 安全相关条目优先保留。
- 不要新增无依据内容。
""".strip()
        res = self.llm_client.create_completion(system_prompt, user_prompt)
        self.debug_trace["step_1_2_filtered_facts"] = clean_json_output(res)

    def step_2_intent_analysis(self):
        system_prompt = "你是医学问答交互分析专家。"
        user_prompt = f"""
任务：分析该问题的用户意图与回答风格需求。

用户问题：
{self.query}

输出 JSON：
{{
  "user_persona": "...",
  "missing_context_questions": ["..."],
  "tone": "..."
}}
""".strip()
        res = self.llm_client.create_completion(system_prompt, user_prompt)
        self.debug_trace["step_2_interaction_intent"] = clean_json_output(res)

    def step_3_synthesis(self):
        system_prompt = "你是资深医学 Rubric 设计专家。"
        evidence_block = self.evidence_text or "（暂无外部检索证据）"
        user_prompt = f"""
任务：基于给定信息，生成用于评估医疗回答质量的 Rubrics（中文）。

用户问题：
{self.query}

检索证据（RAG）：
{evidence_block}

原子事实（未过滤）：
{json.dumps(self.debug_trace.get("step_1_1_atomic_facts", {}), ensure_ascii=False)}

交互意图：
{json.dumps(self.debug_trace.get("step_2_interaction_intent", {}), ensure_ascii=False)}

输出约束：
1) 只输出 JSON，格式：
{{
  "rubrics": [
    {{"criterion": "...", "axis": "...", "points": 0}}
  ]
}}
2) axis 是：
   accuracy, completeness, context_awareness, communication_quality, instruction_following
3) points 必须是 -10 到 10 的整数。
4) criterion 用中文、可执行、可判定，避免空话。
5) 优先覆盖以下类型的医学要点（按问题相关性选择）：
   - 病因/机制解释
   - 鉴别诊断或关键排除项
   - 检查建议与结果解读要点
   - 治疗/用药建议（含禁忌与不良反应）
   - 风险信号与就医触发条件
6) 禁止抽象口号式条目（如仅说“内容全面、表达清晰”而无具体医学检查点或者有用信息）。
7) 条目数建议 8-18。
""".strip()
        res = self.llm_client.create_completion(system_prompt, user_prompt)
        self.debug_trace["step_3_draft_rubrics"] = extract_rubrics_dict(clean_json_output(res))

    def step_4_refinement(self):
        system_prompt = "你是医学 Rubric 终审专家。"
        evidence_block = self.evidence_text or "（暂无外部检索证据）"
        user_prompt = f"""
任务：审计并精炼 Rubrics，补齐遗漏，去重合并，输出最终版本。

用户问题：
{self.query}

检索证据（RAG）：
{evidence_block}

事实依据：
{json.dumps(self.debug_trace.get("step_1_1_atomic_facts", {}), ensure_ascii=False)}

意图分析：
{json.dumps(self.debug_trace.get("step_2_interaction_intent", {}), ensure_ascii=False)}

草稿 Rubrics：
{json.dumps(self.debug_trace.get("step_3_draft_rubrics", {}), ensure_ascii=False)}

输出要求：
1) 只输出 JSON：
{{
  "rubrics": [
    {{"criterion": "...", "axis": "...", "points": 0}}
  ]
}}
2) 保持与事实一致，不引入无依据条目。
3) 删除空泛条目：没有具体医学检查点或有用信息和建议必须删除。
4) 删除重复条目：语义相同只保留更具体、可验证的一条。
""".strip()
        res = self.llm_client.create_completion(system_prompt, user_prompt)
        self.debug_trace["step_4_final_rubrics"] = extract_rubrics_dict(clean_json_output(res))

    def execute(self) -> Dict[str, Any]:
        self.llm_client.reset_metrics()
        self._run_step("step_1_1_atomic_extraction", self.step_1_1_atomic_extraction)
        # 直连模式：跳过 Step 1.2，避免过滤导致信息丢失；为兼容保留字段镜像
        self.debug_trace["step_1_2_filtered_facts"] = self.debug_trace.get("step_1_1_atomic_facts", {})
        self.debug_trace["metrics"]["steps"]["step_1_2_filtering"] = {
            "latency_ms": 0.0,
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "errors": [],
            "skipped": True,
            "reason": "bypassed_to_preserve_recall",
        }
        self._run_step("step_2_intent_analysis", self.step_2_intent_analysis)
        self._run_step("step_3_synthesis", self.step_3_synthesis)
        self._run_step("step_4_refinement", self.step_4_refinement)

        # Fallback: if step_4 invalid/empty, use step_3
        final_obj = self.debug_trace.get("step_4_final_rubrics", {})
        if not isinstance(final_obj, dict) or not isinstance(final_obj.get("rubrics"), list) or not final_obj.get("rubrics"):
            self.debug_trace["step_4_final_rubrics"] = self.debug_trace.get(
                "step_3_draft_rubrics", {"rubrics": []}
            )

        # Normalize final format
        self.debug_trace["step_4_final_rubrics"] = extract_rubrics_dict(
            self.debug_trace["step_4_final_rubrics"]
        )
        self.debug_trace["step_3_draft_rubrics"] = extract_rubrics_dict(
            self.debug_trace["step_3_draft_rubrics"]
        )
        return self.debug_trace

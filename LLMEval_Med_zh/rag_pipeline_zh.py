import os
import json
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from rubrics_pipeline_zh import UniversalLLMClientZH, clean_json_output

try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None

try:
    import trafilatura
except Exception:
    trafilatura = None


class MedicalRAGPipelineZH:
    """
    中文版 Stage 1 Retrieval:
    - Full: Query -> Router(意图+扩展检索词+建议域名) -> Search -> Rerank -> Evidence
    - No Router: Query -> 直接搜索(Verbatim) -> Rerank -> Evidence
    """

    AUTHORITY_DOMAIN_WHITELIST_ZH = [
        # 中国官方/指南
        "nhc.gov.cn",
        "chinacdc.cn",
        "samr.gov.cn",
        "nmpa.gov.cn",
        "gov.cn",
        # 中文医学专业内容
        "cma.org.cn",
        "dxy.cn",
        "medlive.cn",
        "msdmanuals.cn",
        # 国际权威
        "who.int",
        "ncbi.nlm.nih.gov",
        "cdc.gov",
        "nice.org.uk",
        "mayoclinic.org",
    ]

    def __init__(
        self,
        llm_client: UniversalLLMClientZH,
        max_results_per_query: int = 3,
        rerank_top_k: int = 5,
        fetch_full_text: bool = False,
    ):
        if TavilyClient is None:
            raise RuntimeError("缺少 tavily 依赖，请先安装 `tavily-python`。")

        tavily_api_key = (
            os.environ.get("TAVILY_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not tavily_api_key:
            raise RuntimeError("缺少 TAVILY_API_KEY，无法执行 RAG 检索。")

        self.tavily = TavilyClient(api_key=tavily_api_key)
        self.llm_client = llm_client
        self.max_results_per_query = max(1, int(max_results_per_query))
        self.rerank_top_k = max(1, int(rerank_top_k))
        self.fetch_full_text = bool(fetch_full_text)
        self.last_run_metrics: Dict[str, Any] = {}

    def _empty_run_metrics(self) -> Dict[str, Any]:
        return {
            "stages": {},
            "llm_calls": [],
            "totals": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "latency_ms": 0.0,
            },
        }

    def _record_stage(
        self,
        metrics: Dict[str, Any],
        stage_name: str,
        elapsed_ms: float,
        llm_calls: Optional[List[Dict[str, Any]]] = None,
        skipped: bool = False,
        reason: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        calls = llm_calls or []
        prompt_tokens = sum(int(c.get("prompt_tokens", 0)) for c in calls)
        completion_tokens = sum(int(c.get("completion_tokens", 0)) for c in calls)
        total_tokens = sum(int(c.get("total_tokens", 0)) for c in calls)
        errors = [c.get("error") for c in calls if c.get("error")]

        stage_obj = {
            "latency_ms": round(float(elapsed_ms), 2),
            "llm_calls": len(calls),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "errors": errors,
        }
        if skipped:
            stage_obj["skipped"] = True
        if reason:
            stage_obj["reason"] = reason
        if isinstance(extra, dict):
            stage_obj.update(extra)
        metrics["stages"][stage_name] = stage_obj

        metrics["totals"]["prompt_tokens"] += prompt_tokens
        metrics["totals"]["completion_tokens"] += completion_tokens
        metrics["totals"]["total_tokens"] += total_tokens
        metrics["totals"]["latency_ms"] += round(float(elapsed_ms), 2)

        for c in calls:
            cc = dict(c)
            cc["stage"] = stage_name
            metrics["llm_calls"].append(cc)

    @staticmethod
    def _extract_domain(url: str) -> str:
        try:
            host = urlparse(url).netloc.lower()
            if host.startswith("www."):
                host = host[4:]
            return host
        except Exception:
            return ""

    def _normalize_domains(self, domains: Any) -> List[str]:
        if not isinstance(domains, list):
            return []
        allowed = set(self.AUTHORITY_DOMAIN_WHITELIST_ZH)
        out: List[str] = []
        for d in domains:
            ds = str(d).strip().lower()
            if not ds:
                continue
            if ds.startswith("site:"):
                ds = ds[5:].strip()
            if ds in allowed and ds not in out:
                out.append(ds)
        return out

    def _route_query(self, user_query: str) -> Dict[str, Any]:
        system_prompt = "你是医疗检索路由专家。"
        user_prompt = f"""
任务：分析医疗问题，并为中文医疗检索生成高质量计划。

用户问题：
{user_query}

可用权威域名（只可从下列选择）：
{json.dumps(self.AUTHORITY_DOMAIN_WHITELIST_ZH, ensure_ascii=False)}

要求：
1) 给出 intent（如：诊断鉴别、治疗方案、用药安全、急症分诊、检查解读等）。
2) 生成 3-5 条检索词（中文为主，可混合英文医学术语）。
3) 选择 3-8 个最相关域名用于检索。
4) 检索词应具体，包含关键症状/病名/检查/治疗关键词。

输出 JSON：
{{
  "intent": "...",
  "queries": ["..."],
  "target_domains": ["..."]
}}
""".strip()
        text = self.llm_client.create_completion(system_prompt, user_prompt)
        obj = clean_json_output(text)

        intent = str(obj.get("intent", "")).strip() if isinstance(obj, dict) else ""
        queries_raw = obj.get("queries") if isinstance(obj, dict) else None
        domains_raw = obj.get("target_domains") if isinstance(obj, dict) else None

        queries: List[str] = []
        if isinstance(queries_raw, list):
            for q in queries_raw:
                qs = str(q).strip()
                if qs and qs not in queries:
                    queries.append(qs)
        queries = queries[:5]
        if not queries:
            queries = [user_query.strip()]

        domains = self._normalize_domains(domains_raw)
        if not domains:
            domains = list(self.AUTHORITY_DOMAIN_WHITELIST_ZH)

        return {
            "intent": intent or "General",
            "queries": queries,
            "target_domains": domains,
        }

    def _search_web(self, queries: List[str], include_domains: List[str]) -> List[Dict[str, Any]]:
        all_rows: List[Dict[str, Any]] = []
        for q in queries:
            q = str(q).strip()
            if not q:
                continue
            search_kwargs = {
                "query": q,
                "search_depth": "basic",
                "max_results": self.max_results_per_query,
                "include_domains": include_domains,
            }
            try:
                resp = self.tavily.search(**search_kwargs)
            except Exception:
                continue
            for r in (resp or {}).get("results", []) or []:
                url = str(r.get("url", "")).strip()
                if not url:
                    continue
                all_rows.append(
                    {
                        "query": q,
                        "title": str(r.get("title", "")).strip() or "No Title",
                        "href": url,
                        "domain": self._extract_domain(url),
                        "snippet": str(r.get("content", "")).strip(),
                    }
                )

        seen = set()
        uniq: List[Dict[str, Any]] = []
        for row in all_rows:
            u = row.get("href", "")
            if not u or u in seen:
                continue
            seen.add(u)
            uniq.append(row)
        return uniq

    def _parse_indices(self, obj: Any) -> List[int]:
        if isinstance(obj, list):
            arr = obj
        elif isinstance(obj, dict):
            arr = obj.get("indices", [])
            if not isinstance(arr, list):
                arr = []
        elif isinstance(obj, str):
            m = re.search(r"\[[\s\d,]+\]", obj)
            if not m:
                return []
            try:
                arr = json.loads(m.group(0))
            except Exception:
                arr = []
        else:
            arr = []
        out: List[int] = []
        for i in arr:
            try:
                iv = int(i)
            except Exception:
                continue
            if iv not in out:
                out.append(iv)
        return out

    def _rerank_results(self, user_query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(candidates) <= self.rerank_top_k:
            return candidates

        lines: List[str] = []
        for i, c in enumerate(candidates):
            snippet = str(c.get("snippet", "")).replace("\n", " ").strip()
            if len(snippet) > 500:
                snippet = snippet[:500]
            lines.append(
                f"[{i}] URL: {c.get('href','')}\n"
                f"Title: {c.get('title','')}\n"
                f"Snippet: {snippet}\n"
            )
        candidate_text = "\n".join(lines)

        system_prompt = "你是医疗检索结果排序器。"
        user_prompt = f"""
任务：从候选结果中选择最有助于回答用户问题的 Top {self.rerank_top_k} 条。

用户问题：
{user_query}

候选结果：
{candidate_text}

排序标准：
1) 相关性：直接回答问题中的核心医学点。
2) 权威性：优先官方/指南/高质量医学来源。
3) 可用性：包含明确事实、诊疗建议、禁忌或风险提示。
4) 多样性：避免重复同类来源。

只输出 JSON：
{{
  "indices": [0, 2, 5]
}}
""".strip()

        raw = self.llm_client.create_completion(system_prompt, user_prompt)
        parsed = clean_json_output(raw)
        indices = self._parse_indices(parsed if parsed else raw)

        valid: List[int] = []
        for i in indices:
            if 0 <= i < len(candidates) and i not in valid:
                valid.append(i)

        if len(valid) < self.rerank_top_k:
            for i in range(len(candidates)):
                if i not in valid:
                    valid.append(i)
                if len(valid) >= self.rerank_top_k:
                    break
        valid = valid[: self.rerank_top_k]
        return [candidates[i] for i in valid]

    def _extract_page_text(self, url: str, fallback_snippet: str) -> str:
        if not self.fetch_full_text or trafilatura is None:
            return fallback_snippet
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return fallback_snippet
            full_text = trafilatura.extract(
                downloaded, include_comments=False, include_tables=True
            )
            if not full_text:
                return fallback_snippet
            full_text = full_text.strip()
            if len(full_text) > 5000:
                full_text = full_text[:5000]
            return full_text
        except Exception:
            return fallback_snippet

    def _synthesize_evidence(
        self,
        query: str,
        intent: str,
        queries_used: List[str],
        selected_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        raw_sources: List[Dict[str, Any]] = []
        raw_blocks: List[str] = []

        for i, item in enumerate(selected_results, 1):
            url = str(item.get("href", "")).strip()
            title = str(item.get("title", "")).strip() or "Unknown Source"
            domain = str(item.get("domain", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            excerpt = self._extract_page_text(url, snippet)
            raw_sources.append(
                {
                    "source_name": title,
                    "url": url,
                    "domain": domain,
                    "snippet": snippet,
                    "excerpt": excerpt,
                }
            )
            raw_blocks.append(
                f"[Source {i}] {title}\nURL: {url}\nDomain: {domain}\nExcerpt:\n{excerpt}\n"
            )

        system_prompt = "你是医疗证据综合专家。"
        user_prompt = f"""
任务：把检索结果整理为结构化 Evidence Block（用于后续 Rubric 生成）。

用户问题：
{query}

检索词：
{json.dumps(queries_used, ensure_ascii=False)}

候选证据：
{chr(10).join(raw_blocks)}

输出要求（只输出 JSON）：
{{
  "intent_category": "...",
  "search_queries_used": ["..."],
  "evidence_sources": [
    {{
      "source_name": "...",
      "url": "...",
      "relevance": "High/Medium/Low",
      "key_excerpt": "...",
      "conflict_note": null
    }}
  ],
  "synthesis": {{
    "consensus": "...",
    "contention": null,
    "red_flags": ["..."],
    "regional_context": null
  }}
}}

规则：
1) 只引用给定证据，不可编造来源。
2) key_excerpt 要尽量具体，包含可判定医学要点。
3) 若来源存在冲突，在 contention/conflict_note 中说明。
4) red_flags 写需要及时就医或禁忌相关事项。
5) search_queries_used 原样保留输入检索词。
""".strip()

        text = self.llm_client.create_completion(system_prompt, user_prompt)
        obj = clean_json_output(text)
        if not isinstance(obj, dict):
            obj = {}

        evidence_sources = obj.get("evidence_sources")
        if not isinstance(evidence_sources, list) or not evidence_sources:
            evidence_sources = []
            for s in raw_sources:
                evidence_sources.append(
                    {
                        "source_name": s.get("source_name") or s.get("domain") or "Unknown",
                        "url": s.get("url", ""),
                        "relevance": "Medium",
                        "key_excerpt": s.get("excerpt", "") or s.get("snippet", ""),
                        "conflict_note": None,
                    }
                )

        normalized_sources: List[Dict[str, Any]] = []
        for s in evidence_sources:
            if not isinstance(s, dict):
                continue
            source_name = str(s.get("source_name", "")).strip() or "Unknown Source"
            url = str(s.get("url", "")).strip()
            relevance = str(s.get("relevance", "Medium")).strip() or "Medium"
            key_excerpt = str(s.get("key_excerpt", "")).strip()
            if not key_excerpt:
                key_excerpt = str(s.get("snippet", "")).strip()
            conflict_note = s.get("conflict_note")
            if conflict_note is not None:
                conflict_note = str(conflict_note).strip() or None
            if not url:
                continue
            normalized_sources.append(
                {
                    "source_name": source_name,
                    "url": url,
                    "relevance": relevance,
                    "key_excerpt": key_excerpt,
                    "conflict_note": conflict_note,
                }
            )

        syn = obj.get("synthesis")
        if not isinstance(syn, dict):
            syn = {}
        red_flags = syn.get("red_flags")
        if not isinstance(red_flags, list):
            red_flags = []
        red_flags = [str(x).strip() for x in red_flags if str(x).strip()]

        return {
            "intent_category": str(obj.get("intent_category", "")).strip() or intent or "General",
            "search_queries_used": [str(q).strip() for q in queries_used if str(q).strip()],
            "evidence_sources": normalized_sources,
            "synthesis": {
                "consensus": str(syn.get("consensus", "")).strip(),
                "contention": (
                    str(syn.get("contention", "")).strip() or None
                    if syn.get("contention") is not None
                    else None
                ),
                "red_flags": red_flags,
                "regional_context": (
                    str(syn.get("regional_context", "")).strip() or None
                    if syn.get("regional_context") is not None
                    else None
                ),
            },
        }

    def run(self, query: str, no_router: bool = False) -> Dict[str, Any]:
        run_start = time.perf_counter()
        metrics = self._empty_run_metrics()
        q = str(query or "").strip()
        if not q:
            result = {
                "error": "empty_query",
                "intent_category": "General",
                "search_queries_used": [],
                "evidence_sources": [],
                "synthesis": {
                    "consensus": "",
                    "contention": None,
                    "red_flags": [],
                    "regional_context": None,
                },
            }
            metrics["totals"]["latency_ms"] = round((time.perf_counter() - run_start) * 1000, 2)
            result["retrieval_metrics"] = metrics
            self.last_run_metrics = metrics
            return result

        if no_router:
            intent = "General"
            queries = [q]
            domains = list(self.AUTHORITY_DOMAIN_WHITELIST_ZH)
            self._record_stage(
                metrics,
                "step_1_router_and_query",
                elapsed_ms=0.0,
                llm_calls=[],
                skipped=True,
                reason="no_router_mode",
                extra={"queries_count": 1, "domains_count": len(domains)},
            )
        else:
            calls_before = len(self.llm_client.call_history)
            t0 = time.perf_counter()
            plan = self._route_query(q)
            elapsed = (time.perf_counter() - t0) * 1000
            new_calls = self.llm_client.call_history[calls_before:]
            self._record_stage(
                metrics,
                "step_1_router_and_query",
                elapsed_ms=elapsed,
                llm_calls=new_calls,
            )
            intent = str(plan.get("intent", "General")).strip() or "General"
            queries = plan.get("queries", [q]) if isinstance(plan, dict) else [q]
            domains = (
                plan.get("target_domains", list(self.AUTHORITY_DOMAIN_WHITELIST_ZH))
                if isinstance(plan, dict)
                else list(self.AUTHORITY_DOMAIN_WHITELIST_ZH)
            )
            if not isinstance(queries, list) or not queries:
                queries = [q]
            if not isinstance(domains, list) or not domains:
                domains = list(self.AUTHORITY_DOMAIN_WHITELIST_ZH)

        t1 = time.perf_counter()
        candidates = self._search_web(queries=queries, include_domains=domains)
        elapsed = (time.perf_counter() - t1) * 1000
        self._record_stage(
            metrics,
            "step_2_search_web",
            elapsed_ms=elapsed,
            llm_calls=[],
            extra={
                "queries_count": len(queries),
                "candidate_count": len(candidates),
                "domains_count": len(domains),
            },
        )
        if not candidates:
            result = {
                "intent_category": intent,
                "search_queries_used": queries,
                "evidence_sources": [],
                "synthesis": {
                    "consensus": "",
                    "contention": "No retrieval results found.",
                    "red_flags": [],
                    "regional_context": None,
                },
                "retrieval_meta": {
                    "router_used": not no_router,
                    "domains_used": domains,
                    "candidate_count": 0,
                    "selected_count": 0,
                },
            }
            metrics["totals"]["latency_ms"] = round((time.perf_counter() - run_start) * 1000, 2)
            result["retrieval_metrics"] = metrics
            self.last_run_metrics = metrics
            return result

        calls_before = len(self.llm_client.call_history)
        t2 = time.perf_counter()
        selected = self._rerank_results(user_query=q, candidates=candidates)
        elapsed = (time.perf_counter() - t2) * 1000
        new_calls = self.llm_client.call_history[calls_before:]
        self._record_stage(
            metrics,
            "step_3_rerank_results",
            elapsed_ms=elapsed,
            llm_calls=new_calls,
            extra={
                "candidate_count": len(candidates),
                "selected_count": len(selected),
            },
        )

        calls_before = len(self.llm_client.call_history)
        t3 = time.perf_counter()
        evidence = self._synthesize_evidence(
            query=q,
            intent=intent,
            queries_used=queries,
            selected_results=selected,
        )
        elapsed = (time.perf_counter() - t3) * 1000
        new_calls = self.llm_client.call_history[calls_before:]
        self._record_stage(
            metrics,
            "step_4_synthesize_evidence",
            elapsed_ms=elapsed,
            llm_calls=new_calls,
        )

        metrics["totals"]["latency_ms"] = round((time.perf_counter() - run_start) * 1000, 2)
        evidence["retrieval_meta"] = {
            "router_used": not no_router,
            "domains_used": domains,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
        }
        evidence["retrieval_metrics"] = metrics
        self.last_run_metrics = metrics
        return evidence

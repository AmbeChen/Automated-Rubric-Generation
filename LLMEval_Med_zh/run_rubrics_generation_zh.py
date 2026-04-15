import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from rubrics_pipeline_zh import RubricPipelineZH, UniversalLLMClientZH
from rag_pipeline_zh import MedicalRAGPipelineZH


QUOTA_ERROR_KEYWORDS = (
    "error code",
    "tokens per day (TPD)",
    "exceeded your current quota",
    "Tokens per day limit exceeded",
)


def is_quota_error_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return any(k in lowered for k in QUOTA_ERROR_KEYWORDS)


def trace_has_quota_error(obj: Any) -> bool:
    if isinstance(obj, dict):
        return any(trace_has_quota_error(v) for v in obj.values())
    if isinstance(obj, list):
        return any(trace_has_quota_error(v) for v in obj)
    if isinstance(obj, str):
        return is_quota_error_text(obj)
    return False


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _checklist_to_rubrics(checklist: List[str]) -> List[Dict[str, Any]]:
    out = []
    for item in checklist:
        txt = str(item).strip()
        if not txt:
            continue
        out.append(
            {
                "criterion": txt,
                "axis": "completeness",
                "points": 6,
            }
        )
    return out


def _extract_generated_rubrics(record_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = record_obj.get("generated_rubrics")
    if isinstance(raw, dict):
        rubrics = raw.get("rubrics", [])
    elif isinstance(raw, list):
        rubrics = raw
    else:
        rubrics = []
    if not isinstance(rubrics, list):
        return []
    out = []
    for x in rubrics:
        if not isinstance(x, dict):
            continue
        c = str(x.get("criterion", "")).strip()
        if not c:
            continue
        out.append(x)
    return out


def _existing_result_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return len(_extract_generated_rubrics(obj)) > 0


def _rebuild_outputs_from_individual(
    individual_dir: Path,
    generated_out: Path,
    reference_out: Path,
):
    rows = []
    for p in sorted(individual_dir.glob("rubric_*.json"), key=lambda x: int(x.stem.split("_")[-1])):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(obj)

    with generated_out.open("w", encoding="utf-8") as f:
        for obj in rows:
            rec = {
                "index": obj.get("index"),
                "problem": obj.get("problem", ""),
                "difficulty": obj.get("difficulty", ""),
                "generated_rubrics": obj.get("generated_rubrics", {"rubrics": []}),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with reference_out.open("w", encoding="utf-8") as f:
        for obj in rows:
            rec = {
                "index": obj.get("index"),
                "problem": obj.get("problem", ""),
                "difficulty": obj.get("difficulty", ""),
                "reference_rubrics": obj.get("reference_rubrics", {"rubrics": []}),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except Exception:
        return 0


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _sum_step_metrics(steps: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_ms": 0.0,
        "llm_calls": 0,
    }
    if not isinstance(steps, dict):
        return out
    for s in steps.values():
        if not isinstance(s, dict):
            continue
        out["prompt_tokens"] += _safe_int(s.get("prompt_tokens", 0))
        out["completion_tokens"] += _safe_int(s.get("completion_tokens", 0))
        out["total_tokens"] += _safe_int(s.get("total_tokens", 0))
        out["latency_ms"] += _safe_float(s.get("latency_ms", 0.0))
        out["llm_calls"] += _safe_int(s.get("llm_calls", 0))
    out["latency_ms"] = round(out["latency_ms"], 2)
    return out


def _build_experiment_record(index: int, problem: str, evidence_obj: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
    rag_metrics = (evidence_obj or {}).get("retrieval_metrics", {})
    rag_stages = rag_metrics.get("stages", {}) if isinstance(rag_metrics, dict) else {}
    rag_totals = rag_metrics.get("totals", {}) if isinstance(rag_metrics, dict) else {}
    if not isinstance(rag_totals, dict):
        rag_totals = {}

    rubric_metrics = (trace or {}).get("metrics", {})
    rubric_steps = rubric_metrics.get("steps", {}) if isinstance(rubric_metrics, dict) else {}
    rubric_totals = _sum_step_metrics(rubric_steps)

    rag_totals_norm = {
        "prompt_tokens": _safe_int(rag_totals.get("prompt_tokens", 0)),
        "completion_tokens": _safe_int(rag_totals.get("completion_tokens", 0)),
        "total_tokens": _safe_int(rag_totals.get("total_tokens", 0)),
        "latency_ms": round(_safe_float(rag_totals.get("latency_ms", 0.0)), 2),
        "llm_calls": _safe_int(len(rag_metrics.get("llm_calls", []))) if isinstance(rag_metrics, dict) else 0,
    }

    sample_totals = {
        "prompt_tokens": rag_totals_norm["prompt_tokens"] + rubric_totals["prompt_tokens"],
        "completion_tokens": rag_totals_norm["completion_tokens"] + rubric_totals["completion_tokens"],
        "total_tokens": rag_totals_norm["total_tokens"] + rubric_totals["total_tokens"],
        "latency_ms": round(rag_totals_norm["latency_ms"] + rubric_totals["latency_ms"], 2),
        "llm_calls": rag_totals_norm["llm_calls"] + rubric_totals["llm_calls"],
    }

    return {
        "index": index,
        "problem": problem,
        "rag": {
            "stages": rag_stages,
            "totals": rag_totals_norm,
            "llm_calls": rag_metrics.get("llm_calls", []) if isinstance(rag_metrics, dict) else [],
        },
        "rubric": {
            "steps": rubric_steps,
            "totals": rubric_totals,
            "llm_calls": rubric_metrics.get("llm_calls", []) if isinstance(rubric_metrics, dict) else [],
        },
        "sample_totals": sample_totals,
    }


def _rebuild_experiment_records_from_individual(individual_dir: Path, out_jsonl: Path, out_summary_json: Path):
    rows: List[Dict[str, Any]] = []
    for p in sorted(individual_dir.glob("rubric_*.json"), key=lambda x: int(x.stem.split("_")[-1])):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        idx = _safe_int(obj.get("index"))
        problem = str(obj.get("problem", ""))
        evidence_obj = obj.get("evidence_data", {}) if isinstance(obj, dict) else {}
        trace = obj.get("full_trace", {}) if isinstance(obj, dict) else {}
        rec = _build_experiment_record(idx, problem, evidence_obj if isinstance(evidence_obj, dict) else {}, trace if isinstance(trace, dict) else {})
        rows.append(rec)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "samples": len(rows),
        "totals": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0.0,
            "llm_calls": 0,
        },
        "rag_totals": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0.0,
            "llm_calls": 0,
        },
        "rubric_totals": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0.0,
            "llm_calls": 0,
        },
    }

    for r in rows:
        st = r.get("sample_totals", {})
        rt = r.get("rag", {}).get("totals", {})
        bt = r.get("rubric", {}).get("totals", {})
        for key in ["prompt_tokens", "completion_tokens", "total_tokens", "llm_calls"]:
            summary["totals"][key] += _safe_int(st.get(key, 0))
            summary["rag_totals"][key] += _safe_int(rt.get(key, 0))
            summary["rubric_totals"][key] += _safe_int(bt.get(key, 0))
        summary["totals"]["latency_ms"] += _safe_float(st.get("latency_ms", 0.0))
        summary["rag_totals"]["latency_ms"] += _safe_float(rt.get("latency_ms", 0.0))
        summary["rubric_totals"]["latency_ms"] += _safe_float(bt.get("latency_ms", 0.0))

    summary["totals"]["latency_ms"] = round(summary["totals"]["latency_ms"], 2)
    summary["rag_totals"]["latency_ms"] = round(summary["rag_totals"]["latency_ms"], 2)
    summary["rubric_totals"]["latency_ms"] = round(summary["rubric_totals"]["latency_ms"], 2)

    n = max(1, summary["samples"])
    summary["average_per_sample"] = {
        "prompt_tokens": round(summary["totals"]["prompt_tokens"] / n, 2),
        "completion_tokens": round(summary["totals"]["completion_tokens"] / n, 2),
        "total_tokens": round(summary["totals"]["total_tokens"] / n, 2),
        "latency_ms": round(summary["totals"]["latency_ms"] / n, 2),
        "llm_calls": round(summary["totals"]["llm_calls"] / n, 2),
    }

    out_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_last_error_text(evidence_obj: Dict[str, Any], trace: Dict[str, Any]) -> str:
    # Prefer rubric-stage errors, then retrieval-stage errors.
    rubric_calls = (((trace or {}).get("metrics", {}) or {}).get("llm_calls", []) or [])
    for c in reversed(rubric_calls):
        err = c.get("error")
        if isinstance(err, str) and err.strip():
            return err.strip()

    rag_calls = (((evidence_obj or {}).get("retrieval_metrics", {}) or {}).get("llm_calls", []) or [])
    for c in reversed(rag_calls):
        err = c.get("error")
        if isinstance(err, str) and err.strip():
            return err.strip()
    return ""


def main():
    base_dir = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(base_dir / "problems_checklist_clean.jsonl"))
    ap.add_argument("--output-dir", default=str(base_dir / "outputs"))
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--end-index", type=int, default=10**9)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--use-rag", action="store_true", help="启用 Stage 1 RAG 检索")
    ap.add_argument("--no-router", action="store_true", help="RAG 中跳过路由，直接 verbatim query 检索")
    ap.add_argument("--evidence-dir", default="", help="RAG evidence 缓存目录（默认 output-dir/evidence_zh）")
    ap.add_argument("--fetch-full-text", action="store_true", help="尝试抓取网页全文（较慢）")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    individual_dir = output_dir / "rubrics_individual_zh"
    generated_out = output_dir / "rubrics_generated_zh.jsonl"
    reference_out = output_dir / "reference_rubrics_from_checklist_zh.jsonl"
    experiment_records_out = output_dir / "experiment_records_zh.jsonl"
    experiment_summary_out = output_dir / "experiment_complexity_summary_zh.json"
    evidence_dir = (
        Path(args.evidence_dir)
        if str(args.evidence_dir).strip()
        else (output_dir / "evidence_zh")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir.mkdir(parents=True, exist_ok=True)
    if args.use_rag:
        evidence_dir.mkdir(parents=True, exist_ok=True)

    data = _load_jsonl(input_path)
    print(f"📂 Loaded {len(data)} items from {input_path}")

    try:
        llm_client = UniversalLLMClientZH()
    except Exception as e:
        print(f"❌ LLM client init failed: {e}")
        return

    rag_pipeline = None
    if args.use_rag:
        try:
            rag_pipeline = MedicalRAGPipelineZH(
                llm_client=llm_client,
                fetch_full_text=args.fetch_full_text,
            )
            mode_txt = "No Router" if args.no_router else "Full Router"
            print(f"🔎 RAG enabled ({mode_txt}), evidence cache: {evidence_dir}")
        except Exception as e:
            print(f"❌ RAG pipeline init failed: {e}")
            return

    processed = 0
    skipped_existing = 0
    skipped_empty = 0
    failed = 0
    evidence_new = 0
    evidence_reused = 0

    for idx, row in enumerate(data):
        if idx < args.start_index or idx > args.end_index:
            continue

        problem = str(row.get("problem", "")).strip()
        difficulty = str(row.get("difficulty", "")).strip()
        checklist = row.get("checklist", [])
        if not isinstance(checklist, list):
            checklist = []

        if not problem:
            skipped_empty += 1
            continue

        out_file = individual_dir / f"rubric_{idx}.json"
        if args.resume and _existing_result_valid(out_file):
            skipped_existing += 1
            continue

        print(f"\n🔬 Processing index={idx}: {problem[:60]}...")
        evidence_obj: Dict[str, Any] = {}
        if args.use_rag and rag_pipeline is not None:
            evidence_file = evidence_dir / f"conversation_{idx}.json"
            if args.resume and evidence_file.exists() and evidence_file.stat().st_size > 0:
                evidence_obj = _load_json(evidence_file)
                if evidence_obj:
                    evidence_reused += 1
            if not evidence_obj:
                evidence_obj = rag_pipeline.run(problem, no_router=args.no_router)
                try:
                    evidence_file.write_text(
                        json.dumps(evidence_obj, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    evidence_new += 1
                except Exception as e:
                    print(f"   ⚠️ Evidence save failed: {e}")

            if trace_has_quota_error(evidence_obj):
                print("🛑 Detected API quota/token issue in RAG stage. Stop now.")
                break

        pipeline = RubricPipelineZH(
            query=problem,
            llm_client=llm_client,
            evidence_data=evidence_obj,
        )
        trace = pipeline.execute()

        if trace_has_quota_error(trace):
            print("🛑 Detected API quota/token issue in trace. Stop now.")
            break

        generated_rubrics = trace.get("step_4_final_rubrics") or trace.get("step_3_draft_rubrics") or {"rubrics": []}
        count = len((generated_rubrics or {}).get("rubrics", [])) if isinstance(generated_rubrics, dict) else 0
        experiment_record = _build_experiment_record(
            index=idx,
            problem=problem,
            evidence_obj=evidence_obj,
            trace=trace,
        )
        record = {
            "index": idx,
            "problem": problem,
            "difficulty": difficulty,
            "source_checklist": checklist,
            "reference_rubrics": {"rubrics": _checklist_to_rubrics(checklist)},
            "generated_rubrics": generated_rubrics,
            "evidence_data": evidence_obj,
            "full_trace": trace,
            "experiment_record": experiment_record,
        }
        try:
            out_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            processed += 1
            print(f"   ✅ Saved {out_file.name} | generated criteria: {count}")
            if count == 0:
                print("🛑 Stop now: generated criteria == 0")
                err_text = _get_last_error_text(evidence_obj, trace)
                if err_text:
                    print(f"   ↳ last error: {err_text}")
                break
        except Exception as e:
            failed += 1
            print(f"   ❌ Save failed: {e}")

    _rebuild_outputs_from_individual(individual_dir, generated_out, reference_out)
    _rebuild_experiment_records_from_individual(
        individual_dir=individual_dir,
        out_jsonl=experiment_records_out,
        out_summary_json=experiment_summary_out,
    )
    print("\n✅ Rebuilt aggregated outputs:")
    print(f"   - {generated_out}")
    print(f"   - {reference_out}")
    print(f"   - {experiment_records_out}")
    print(f"   - {experiment_summary_out}")
    print(
        f"📊 Summary: processed={processed}, skipped_existing={skipped_existing}, "
        f"skipped_empty={skipped_empty}, failed={failed}, "
        f"evidence_new={evidence_new}, evidence_reused={evidence_reused}"
    )


if __name__ == "__main__":
    main()

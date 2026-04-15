import ast
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from openai import OpenAI

from config import API_KEY, API_PROVIDER, BASE_URL, JUDGE_MODEL


# ================= CONFIG =================
START_INDEX = int(os.getenv("START_INDEX", 0))
END_INDEX = int(os.getenv("END_INDEX", 100))

DATA_DIR = "data_evaluation"
GOLD_ATOMIC_PATH = "data_evaluation/gold_atomic_points_all.jsonl"
GEN_RUBRIC_PATHS = {
    "full": os.path.join(DATA_DIR, "rubrics_full.jsonl"),
    "no_router": os.path.join(DATA_DIR, "rubrics_no_router.jsonl"),
    "no_atomic": os.path.join(DATA_DIR, "rubrics_no_atomic.jsonl"),
    "no_intent": os.path.join(DATA_DIR, "rubrics_no_intent.jsonl"),
    "no_audit": os.path.join(DATA_DIR, "rubrics_no_audit.jsonl"),
    # Direct baseline rubrics from GPT-4o conversion
    "gpt4o": "data/rubrics_GPT4o_converted.jsonl",
    # Generic baseline rubrics (shared criteria list)
    "generic": "data/generic_rubrics.jsonl",
}

CACHE_ROOT = os.getenv("CIA_CACHE_ROOT", "outputs/ablation_cia_atomic_cache")
SUMMARY_PATH = os.getenv("CIA_SUMMARY_PATH", "outputs/ablation_cia_atomic_summary.json")
_default_compact = (
    SUMMARY_PATH[:-5] + "_compact.json" if SUMMARY_PATH.lower().endswith(".json") else SUMMARY_PATH + "_compact.json"
)
COMPACT_SUMMARY_PATH = os.getenv("CIA_COMPACT_SUMMARY_PATH", _default_compact)
P_VALUE_BASELINE_MODE = os.getenv("P_VALUE_BASELINE_MODE", "gpt4o")

MAX_RETRY = int(os.getenv("CIA_MAX_RETRY", 3))
RETRY_SLEEP = float(os.getenv("CIA_RETRY_SLEEP", 2.0))

QUOTA_ERROR_KEYWORDS = (
    "error code",
    "insufficient",
    "quota",
    "rate limit",
    "429",
    "billing",
    "credit",
    "exceeded your current quota",
    "token",
    "quota_exhausted",
)


client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_modes() -> List[str]:
    raw = os.getenv("CIA_MODES", "").strip()
    default_modes = list(GEN_RUBRIC_PATHS.keys())
    if not raw:
        return default_modes

    out: List[str] = []
    for part in raw.split(","):
        mode = part.strip()
        if not mode:
            continue
        if mode not in GEN_RUBRIC_PATHS:
            print(f"⚠️ Unknown mode [{mode}] in CIA_MODES, ignoring.")
            continue
        if mode not in out:
            out.append(mode)
    return out or default_modes


MODES = _parse_modes()


def _is_quota_error_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return any(k in lowered for k in QUOTA_ERROR_KEYWORDS)


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


def _safe_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text or "")
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def _normalize_generated_rubrics(raw) -> List[str]:
    if isinstance(raw, dict):
        rubrics = raw.get("rubrics", [])
    elif isinstance(raw, list):
        rubrics = raw
    else:
        rubrics = []

    texts = []
    for r in rubrics:
        if not isinstance(r, dict):
            continue
        t = str(r.get("criterion", "")).strip()
        if t:
            texts.append(t)
    return texts


def _extract_criteria_from_list(items) -> List[str]:
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        t = str(item.get("criterion", "")).strip()
        if t:
            out.append(t)
    return out


def _load_generic_criteria(path: str) -> List[str]:
    # 1) Try parse whole file as JSON object/list
    try:
        text = open(path, "r", encoding="utf-8").read().strip()
    except Exception:
        return []
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return _extract_criteria_from_list(parsed.get("rubrics", []))
        if isinstance(parsed, list):
            return _extract_criteria_from_list(parsed)
    except Exception:
        pass

    # 2) Try Python literal list/dict (handles trailing commas in this file)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return _extract_criteria_from_list(parsed.get("rubrics", []))
        if isinstance(parsed, list):
            return _extract_criteria_from_list(parsed)
    except Exception:
        pass

    # 3) Fallback: line-by-line jsonl
    criteria = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            t = str(obj.get("criterion", "")).strip()
            if t:
                criteria.append(t)
    return criteria


def load_generated_rubrics_for_mode(
    mode: str, available_conv_ids: Optional[List[int]] = None
) -> Dict[int, List[str]]:
    path = GEN_RUBRIC_PATHS.get(mode)
    if not path:
        return {}
    if not os.path.exists(path):
        return {}

    if mode == "generic":
        criteria = _load_generic_criteria(path)
        if not criteria:
            return {}
        if available_conv_ids is None:
            conv_ids = list(range(START_INDEX, END_INDEX + 1))
        else:
            conv_ids = sorted(set(int(x) for x in available_conv_ids))
        return {cid: criteria for cid in conv_ids}

    out: Dict[int, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                raw_id = obj.get("index", obj.get("conv_id"))
                if raw_id is None:
                    continue
                conv_id = int(raw_id)
            except Exception:
                continue

            criteria = _normalize_generated_rubrics(
                obj.get("generated_rubrics", obj.get("rubrics", []))
            )
            if criteria:
                out[conv_id] = criteria
    return out


def _wilson_ci_95(yes: int, total: int):
    if total <= 0:
        return None, None
    z = 1.959963984540054
    p = yes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = (
        z
        * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)
        / denom
    )
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return low, high


def _two_proportion_pvalue(x1: int, n1: int, x2: int, n2: int):
    if n1 <= 0 or n2 <= 0:
        return None
    p_pool = (x1 + x2) / (n1 + n2)
    var = p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)
    if var <= 0:
        p1 = x1 / n1
        p2 = x2 / n2
        return 1.0 if abs(p1 - p2) < 1e-12 else 0.0
    z = (x1 / n1 - x2 / n2) / math.sqrt(var)
    # two-sided p-value
    return math.erfc(abs(z) / math.sqrt(2.0))


def _build_compact_summary(full_summary: Dict, p_value_key: str) -> Dict:
    compact_modes = []
    for s in full_summary.get("modes", []):
        compact_modes.append(
            {
                "mode": s.get("mode"),
                "total_gold_atomic_points": s.get("total_gold_atomic_points"),
                "count_yes": s.get("count_yes"),
                "cia_score_percent": s.get("cia_score_percent"),
                "ci_95_percent": s.get("ci_95_percent"),
                p_value_key: s.get(p_value_key),
                "stopped": s.get("stopped", False),
            }
        )

    return {
        "script": full_summary.get("script"),
        "started_at": full_summary.get("started_at"),
        "finished_at": full_summary.get("finished_at"),
        "judge_model": full_summary.get("judge_model"),
        "api_provider": full_summary.get("api_provider"),
        "cache_root": full_summary.get("cache_root"),
        "summary_path": full_summary.get("summary_path"),
        "compact_summary_path": COMPACT_SUMMARY_PATH,
        "gold_atomic_path": full_summary.get("gold_atomic_path"),
        "range": full_summary.get("range"),
        "modes_requested": full_summary.get("modes_requested"),
        "p_value_baseline_mode": full_summary.get("p_value_baseline_mode"),
        "modes": compact_modes,
    }


def load_gold_atomic_points(path: str) -> Dict[int, List[Dict]]:
    out: Dict[int, List[Dict]] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                raw_id = obj.get("conv_id", obj.get("index"))
                if raw_id is None:
                    continue
                conv_id = int(raw_id)
            except Exception:
                continue

            points = obj.get("gold_atomic_points", obj.get("gold_keypoints", []))
            if isinstance(points, list) and points:
                out[conv_id] = points
    return out


def judge_entailment(gen_criteria: List[str], hypothesis: str) -> Dict:
    premise = "\n".join([f"{i + 1}. {c}" for i, c in enumerate(gen_criteria)])
    prompt = f"""
You are doing Asymmetric Verification.

Premise (Generated Rubric criteria list, knowledge source):
{premise}

Hypothesis (single Gold Atomic Point):
{hypothesis}

Task:
Determine if Hypothesis is EXPLICITLY entailed by ANY of the criteria in Premise.

STRICT ADHERENCE REQUIRED:
- Strict Intent Alignment: a Premise criterion MUST explicitly require the exact specific action, fact, or behavior described in the Hypothesis.
- Granularity Rule: broad or generic instructions do NOT entail a specific atomic point unless the specific core intent is explicitly stated.
- No Implicit Assumptions: do not assume the model/doctor would "naturally know" to include a specific detail from a broad category.
- If the core specific intent is missing, output NO.
- When in doubt, choose NO.

If YES, you MUST cite one supporting criterion index from the Premise. 
If no criterion explicitly meets this strict standard, output NO.

Return strict JSON only:
{{
  "decision": "YES" or "NO",
  "support_index": 1,
  "reason": "short justification"
}}

Rules:
- Use only Premise content.
- No partial labels; only YES/NO.
- If decision is NO, set support_index to null.
- Keep reason concise and state why coverage is explicit (YES) or missing (NO).
"""
    last_error = ""
    for attempt in range(MAX_RETRY):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            parsed = _safe_parse_json(raw)
            if not isinstance(parsed, dict):
                last_error = "invalid_json_output"
                continue

            decision = str(parsed.get("decision", "")).strip().upper()
            support_index = parsed.get("support_index")
            reason = str(parsed.get("reason", "")).strip()

            if decision not in ("YES", "NO"):
                last_error = "invalid_decision"
                continue

            if decision == "YES":
                try:
                    support_index = int(support_index)
                except Exception:
                    decision = "NO"
                    support_index = None
                else:
                    if support_index < 1 or support_index > len(gen_criteria):
                        decision = "NO"
                        support_index = None
            else:
                support_index = None

            return {
                "decision": decision,
                "support_index": support_index,
                "reason": reason,
                "raw_output": raw,
            }
        except Exception as e:
            last_error = _extract_error_message(e)
            print(f"[Retry {attempt + 1}] judge error: {last_error}")
            if _is_quota_error_text(last_error):
                raise RuntimeError(last_error) from e
            time.sleep(RETRY_SLEEP + attempt)

    raise RuntimeError(f"Judge failed after retries: {last_error}")


def _build_eval_record(point: Dict, verdict: Dict, gen_criteria: List[str]) -> Dict:
    ga_id = point.get("ga_id", "")
    text = point.get("text", "")
    support_index = verdict.get("support_index")
    support_text = (
        gen_criteria[support_index - 1]
        if isinstance(support_index, int) and 1 <= support_index <= len(gen_criteria)
        else None
    )
    return {
        "ga_id": ga_id,
        "atomic_point": text,
        "decision": verdict.get("decision", "NO"),
        "support_index": support_index,
        "support_criterion": support_text,
        "reason": verdict.get("reason", ""),
        "raw_output": verdict.get("raw_output", ""),
    }


def run_cia_for_mode(mode: str, gold_data: Dict[int, List[Dict]]) -> Optional[Dict]:
    print(f"\n🔬 Atomic CIA for Mode [{mode}] (Range {START_INDEX}-{END_INDEX})")
    mode_started = _utc_now()
    mode_start_ts = datetime.now(timezone.utc).timestamp()
    gen_data = load_generated_rubrics_for_mode(mode, list(gold_data.keys()))
    if not gen_data:
        print(f"   ⚠️ No generated rubrics for mode [{mode}], skipping.")
        return None

    cache_dir = os.path.join(CACHE_ROOT, mode)
    os.makedirs(cache_dir, exist_ok=True)

    total_gold_atomic = 0
    total_yes = 0
    processed_new = 0
    cache_hits = 0
    skipped_missing_generated = 0
    processed_conv_ids: List[int] = []
    cache_hit_conv_ids: List[int] = []
    error_messages: List[str] = []

    for conv_id in sorted(gold_data.keys()):
        if not (START_INDEX <= conv_id <= END_INDEX):
            continue
        if conv_id not in gen_data:
            skipped_missing_generated += 1
            continue

        gold_atomic_points = gold_data[conv_id]
        if not gold_atomic_points:
            continue

        cache_file = os.path.join(cache_dir, f"{conv_id}.json")
        records = []

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as cf:
                    cached = json.load(cf)
                records = cached.get("atomic_results", [])
            except Exception:
                records = []

        if not records:
            gen_criteria = gen_data[conv_id]
            print(f"   ⚡ Evaluating conv_id={conv_id} ...", end="\r")

            for p in gold_atomic_points:
                hypothesis = str(p.get("text", "")).strip()
                if not hypothesis:
                    continue
                try:
                    verdict = judge_entailment(gen_criteria, hypothesis)
                except Exception as e:
                    msg = _extract_error_message(e)
                    print(f"\n   🛑 Judge error at conv_id={conv_id}: {msg}")
                    if msg and len(error_messages) < 50:
                        error_messages.append(msg)
                    if _is_quota_error_text(msg):
                        print("   🛑 API quota/token issue detected. Stopping CIA batch now.")
                        return {
                            "mode": mode,
                            "stopped": True,
                            "message": msg,
                            "processed_new_items": processed_new,
                            "cache_hit_items": cache_hits,
                            "skipped_missing_generated": skipped_missing_generated,
                            "processed_conv_ids": processed_conv_ids,
                            "cache_hit_conv_ids": cache_hit_conv_ids,
                            "error_messages": error_messages,
                            "started_at": mode_started,
                            "finished_at": _utc_now(),
                            "elapsed_seconds": round(
                                datetime.now(timezone.utc).timestamp() - mode_start_ts, 3
                            ),
                        }
                    verdict = {
                        "decision": "NO",
                        "support_index": None,
                        "reason": f"judge_error: {msg}",
                        "raw_output": "",
                    }

                records.append(_build_eval_record(p, verdict, gen_criteria))

            with open(cache_file, "w", encoding="utf-8") as cf:
                json.dump(
                    {
                        "conv_id": conv_id,
                        "mode": mode,
                        "generated_criteria_count": len(gen_data[conv_id]),
                        "gold_atomic_count": len(gold_atomic_points),
                        "atomic_results": records,
                    },
                    cf,
                    indent=2,
                    ensure_ascii=False,
                )
            processed_new += 1
            processed_conv_ids.append(conv_id)
        else:
            cache_hits += 1
            cache_hit_conv_ids.append(conv_id)

        for r in records:
            total_gold_atomic += 1
            if str(r.get("decision", "")).upper() == "YES":
                total_yes += 1

    cia_score = (total_yes / total_gold_atomic * 100) if total_gold_atomic else 0.0
    ci_low, ci_high = _wilson_ci_95(total_yes, total_gold_atomic)
    print(f"\n   📊 Mode [{mode}] Statistics:")
    print(f"      Processed New Items:        {processed_new}")
    print(f"      Cache Hit Items:            {cache_hits}")
    print(f"      Missing Generated Rubrics:  {skipped_missing_generated}")
    print(f"      Total Gold Atomic Points:   {total_gold_atomic}")
    print(f"      Count(YES):                 {total_yes}")
    print(f"      CIA Score (YES/Total):      {cia_score:.2f}%")
    if ci_low is not None and ci_high is not None:
        print(f"      95% CI:                     [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")

    return {
        "mode": mode,
        "processed_new_items": processed_new,
        "cache_hit_items": cache_hits,
        "skipped_missing_generated": skipped_missing_generated,
        "processed_conv_ids": processed_conv_ids,
        "cache_hit_conv_ids": cache_hit_conv_ids,
        "error_messages": error_messages,
        "total_gold_atomic_points": total_gold_atomic,
        "count_yes": total_yes,
        "cia_score_percent": cia_score,
        "ci_95_percent": {
            "low": (ci_low * 100 if ci_low is not None else None),
            "high": (ci_high * 100 if ci_high is not None else None),
        },
        "stopped": False,
        "started_at": mode_started,
        "finished_at": _utc_now(),
        "elapsed_seconds": round(datetime.now(timezone.utc).timestamp() - mode_start_ts, 3),
    }


def main():
    run_started = _utc_now()
    if not os.path.exists(GOLD_ATOMIC_PATH):
        print(f"❌ Gold atomic file not found: {GOLD_ATOMIC_PATH}")
        print("   Run first: python3 eval_pipeline/build_gold_atomic_points.py")
        return

    gold_data = load_gold_atomic_points(GOLD_ATOMIC_PATH)
    if not gold_data:
        print(f"❌ Failed to load gold atomic points from: {GOLD_ATOMIC_PATH}")
        return

    summary_dir = os.path.dirname(SUMMARY_PATH)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)

    summary = {
        "script": "eval_pipeline/eval_cia_batch.py",
        "started_at": run_started,
        "finished_at": None,
        "judge_model": JUDGE_MODEL,
        "api_provider": API_PROVIDER,
        "cache_root": CACHE_ROOT,
        "summary_path": SUMMARY_PATH,
        "gold_atomic_path": GOLD_ATOMIC_PATH,
        "range": {"start": START_INDEX, "end": END_INDEX},
        "modes_requested": MODES,
        "modes": [],
    }

    for mode in MODES:
        stats = run_cia_for_mode(mode, gold_data)
        if stats is None:
            continue
        summary["modes"].append(stats)
        if stats.get("stopped"):
            break

    # p-value: two-sided two-proportion test against baseline mode (default: gpt4o)
    baseline_stats = None
    for s in summary["modes"]:
        if s.get("mode") == P_VALUE_BASELINE_MODE and not s.get("stopped"):
            baseline_stats = s
            break

    p_value_key = f"p_value_vs_{P_VALUE_BASELINE_MODE}"
    summary["p_value_baseline_mode"] = P_VALUE_BASELINE_MODE

    for s in summary["modes"]:
        if s.get("stopped"):
            s[p_value_key] = None
            continue
        if baseline_stats is None:
            s[p_value_key] = None
            continue
        if s.get("mode") == P_VALUE_BASELINE_MODE:
            s[p_value_key] = 1.0
            continue
        s[p_value_key] = _two_proportion_pvalue(
            int(s.get("count_yes", 0)),
            int(s.get("total_gold_atomic_points", 0)),
            int(baseline_stats.get("count_yes", 0)),
            int(baseline_stats.get("total_gold_atomic_points", 0)),
        )

    summary["finished_at"] = _utc_now()
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Summary saved to: {SUMMARY_PATH}")

    compact_summary = _build_compact_summary(summary, p_value_key)
    compact_dir = os.path.dirname(COMPACT_SUMMARY_PATH)
    if compact_dir:
        os.makedirs(compact_dir, exist_ok=True)
    with open(COMPACT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(compact_summary, f, indent=2, ensure_ascii=False)
    print(f"✅ Compact summary saved to: {COMPACT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()

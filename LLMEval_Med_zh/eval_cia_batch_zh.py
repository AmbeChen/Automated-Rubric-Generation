import ast
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI


START_INDEX = int(os.getenv("START_INDEX", 0))
END_INDEX = int(os.getenv("END_INDEX", 50))
MAX_RETRY = int(os.getenv("CIA_MAX_RETRY", 3))
RETRY_SLEEP = float(os.getenv("CIA_RETRY_SLEEP", 2.0))
JUDGE_POLICY = os.getenv("CIA_JUDGE_POLICY", "strict").strip().lower()
if JUDGE_POLICY not in ("strict", "lenient"):
    JUDGE_POLICY = "strict"

BASE_DIR = Path(__file__).resolve().parent
GOLD_ATOMIC_PATH = Path(
    os.getenv(
        "CIA_GOLD_ATOMIC_PATH",
        str(BASE_DIR / "outputs" / "eval" / "gold_atomic_points_zh.jsonl"),
    )
)
default_cache_root = BASE_DIR / "outputs" / "eval" / ("cia_cache" if JUDGE_POLICY == "strict" else f"cia_cache_{JUDGE_POLICY}")
default_summary_path = BASE_DIR / "outputs" / "eval" / ("cia_summary_zh.json" if JUDGE_POLICY == "strict" else f"cia_summary_zh_{JUDGE_POLICY}.json")
CACHE_ROOT = Path(os.getenv("CIA_CACHE_ROOT", str(default_cache_root)))
SUMMARY_PATH = Path(os.getenv("CIA_SUMMARY_PATH", str(default_summary_path)))

MODE_RUBRIC_PATHS = {
    "ours": str(BASE_DIR / "outputs" / "rubrics_generated_zh.jsonl"),
    "generic": str(BASE_DIR / "outputs" / "eval" / "generic_rubrics.jsonl"),
}

# Extend modes by env: CIA_EXTRA_MODE_PATHS="gpt4o=data/rubrics_GPT4o_converted.jsonl,generic=data/generic_rubrics.jsonl"
extra = os.getenv("CIA_EXTRA_MODE_PATHS", "").strip()
if extra:
    for pair in extra.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k and v:
            MODE_RUBRIC_PATHS[k] = v

raw_modes = os.getenv("CIA_MODES", ",".join(MODE_RUBRIC_PATHS.keys()))
MODES = [m.strip() for m in raw_modes.split(",") if m.strip() in MODE_RUBRIC_PATHS]
if not MODES:
    MODES = list(MODE_RUBRIC_PATHS.keys())

P_VALUE_BASELINE_MODE = os.getenv("P_VALUE_BASELINE_MODE", "ours")

API_PROVIDER = os.getenv("LLM_PROVIDER", "cerebras").lower()
if API_PROVIDER == "groq":
    API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    BASE_URL = "https://api.groq.com/openai/v1"
elif API_PROVIDER == "cerebras":
    API_KEY = os.getenv("CEREBRAS_API_KEY") or os.getenv("OPENAI_API_KEY")
    BASE_URL = "https://api.cerebras.ai/v1"
else:
    API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = "https://api.openai.com/v1"

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")
TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))

QUOTA_ERROR_KEYWORDS = (
    "error code",
    "judge_error",
    "rate limit",
    "exceeded your current quota",
    "tokens per day limit exceeded"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_quota_error_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return any(k in lowered for k in QUOTA_ERROR_KEYWORDS)


def _extract_error_message(err: Exception) -> str:
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
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _extract_criteria(items) -> List[str]:
    if not isinstance(items, list):
        return []
    out = []
    for x in items:
        if not isinstance(x, dict):
            continue
        t = str(x.get("criterion", "")).strip()
        if t:
            out.append(t)
    return out


def _load_generic_criteria(path: str) -> List[str]:
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except Exception:
        return []
    if not text:
        return []
    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        pass
    if parsed is None:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
    if isinstance(parsed, dict):
        return _extract_criteria(parsed.get("rubrics", []))
    if isinstance(parsed, list):
        return _extract_criteria(parsed)

    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            t = str(obj.get("criterion", "")).strip()
            if t:
                out.append(t)
    return out


def load_generated_rubrics(path: str, available_ids: Optional[List[int]] = None, mode: Optional[str] = None) -> Dict[int, List[str]]:
    p = Path(path)
    if not p.exists():
        return {}

    if (mode or "").strip().lower() == "generic" or p.name == "generic_rubrics.jsonl":
        criteria = _load_generic_criteria(path)
        if not criteria:
            return {}
        ids = sorted(set(available_ids or []))
        if not ids:
            ids = list(range(START_INDEX, END_INDEX + 1))
        return {i: criteria for i in ids}

    out: Dict[int, List[str]] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            raw_id = obj.get("index", obj.get("conv_id"))
            try:
                idx = int(raw_id)
            except Exception:
                continue
            raw = obj.get("generated_rubrics", obj.get("rubrics", []))
            if isinstance(raw, dict):
                criteria = _extract_criteria(raw.get("rubrics", []))
            else:
                criteria = _extract_criteria(raw)
            if criteria:
                out[idx] = criteria
    return out


def load_gold_atomic(path: Path) -> Dict[int, List[Dict]]:
    out: Dict[int, List[Dict]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            try:
                idx = int(obj.get("index", obj.get("conv_id")))
            except Exception:
                continue
            pts = obj.get("gold_atomic_points", [])
            if isinstance(pts, list) and pts:
                out[idx] = pts
    return out


def _wilson_ci_95(yes: int, total: int):
    if total <= 0:
        return None, None
    z = 1.959963984540054
    p = yes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _two_proportion_pvalue(x1: int, n1: int, x2: int, n2: int):
    if n1 <= 0 or n2 <= 0:
        return None
    p_pool = (x1 + x2) / (n1 + n2)
    var = p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)
    if var <= 0:
        return 1.0
    z = (x1 / n1 - x2 / n2) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2.0))


def _build_judge_prompt(gen_criteria: List[str], hypothesis: str) -> str:
    premise = "\n".join([f"{i + 1}. {c}" for i, c in enumerate(gen_criteria)])
    if JUDGE_POLICY == "lenient":
        return f"""
你在做 Asymmetric Verification。

Premise（生成 rubric 的 criteria 列表）：
{premise}

Hypothesis（单条 gold atomic point）：
{hypothesis}

任务：
1. 判断 Premise 在逻辑上是否包含 Hypothesis。
2. 如果 YES，必须在Premise中给出支持条目的 index。
3. 如果没有明确支持的条件，则输出no。

仅输出 JSON：
{{
  "decision": "YES" or "NO",
  "support_index": 1,
  "reason": "..."
}}
若 NO，support_index 设为 null。
""".strip()

    return f"""
你在做 Asymmetric Verification。

Premise（生成 rubric 的 criteria 列表）：
{premise}

Hypothesis（单条 gold atomic point）：
{hypothesis}

任务：
判断 Premise 是否“明确蕴含” Hypothesis。

严格规则：
1) 只有明确覆盖核心意图才算 YES。
2) 宽泛/笼统表述不算覆盖具体 atomic point。
3) 不允许基于常识进行隐含推断。
4) 若无法明确对应，输出 NO。
5) 若 YES，必须给出支持条目的 index（1-based）。

仅输出 JSON：
{{
  "decision": "YES" or "NO",
  "support_index": 1,
  "reason": "..."
}}
若 NO，support_index 设为 null。
""".strip()


def judge_entailment(client: OpenAI, gen_criteria: List[str], hypothesis: str) -> Dict:
    prompt = _build_judge_prompt(gen_criteria, hypothesis)

    last_error = ""
    for attempt in range(MAX_RETRY):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
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
            if _is_quota_error_text(last_error):
                raise RuntimeError(last_error) from e
            time.sleep(RETRY_SLEEP + attempt)
    raise RuntimeError(last_error or "judge_failed")


def run_mode(client: OpenAI, mode: str, gold_data: Dict[int, List[Dict]]) -> Optional[Dict]:
    path = MODE_RUBRIC_PATHS.get(mode)
    if not path:
        return None
    gen = load_generated_rubrics(path, list(gold_data.keys()), mode=mode)
    if not gen:
        print(f"⚠️ mode={mode} has no generated rubrics, skip.")
        return None

    cache_dir = CACHE_ROOT / mode
    cache_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    yes = 0
    processed_new = 0
    cache_hit = 0
    skipped_missing = 0
    error_messages: List[str] = []
    stopped = False
    stop_message = ""
    started = _utc_now()
    t0 = datetime.now(timezone.utc).timestamp()

    for idx in sorted(gold_data.keys()):
        if not (START_INDEX <= idx <= END_INDEX):
            continue
        if idx not in gen:
            skipped_missing += 1
            continue

        points = gold_data[idx]
        if not points:
            continue

        cache_file = cache_dir / f"{idx}.json"
        atomic_results = []
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                cached_policy = str(cached.get("judge_policy", "strict")).strip().lower()
                if cached_policy == JUDGE_POLICY:
                    atomic_results = cached.get("atomic_results", [])
                else:
                    atomic_results = []
            except Exception:
                atomic_results = []

        if not atomic_results:
            criteria = gen[idx]
            for p in points:
                hypothesis = str(p.get("text", "")).strip()
                if not hypothesis:
                    continue
                try:
                    verdict = judge_entailment(client, criteria, hypothesis)
                except Exception as e:
                    msg = _extract_error_message(e)
                    if msg and len(error_messages) < 50:
                        error_messages.append(msg)
                    if _is_quota_error_text(msg):
                        stopped = True
                        stop_message = msg
                        break
                    verdict = {
                        "decision": "NO",
                        "support_index": None,
                        "reason": f"judge_error: {msg}",
                        "raw_output": "",
                    }

                support_index = verdict.get("support_index")
                support_text = (
                    criteria[support_index - 1]
                    if isinstance(support_index, int) and 1 <= support_index <= len(criteria)
                    else None
                )
                atomic_results.append(
                    {
                        "ga_id": p.get("ga_id", ""),
                        "atomic_point": hypothesis,
                        "decision": verdict.get("decision", "NO"),
                        "support_index": support_index,
                        "support_criterion": support_text,
                        "reason": verdict.get("reason", ""),
                        "raw_output": verdict.get("raw_output", ""),
                    }
                )

            cache_file.write_text(
                json.dumps(
                    {
                        "index": idx,
                        "mode": mode,
                        "judge_policy": JUDGE_POLICY,
                        "generated_criteria_count": len(gen[idx]),
                        "gold_atomic_count": len(points),
                        "atomic_results": atomic_results,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            processed_new += 1
        else:
            cache_hit += 1

        for r in atomic_results:
            total += 1
            if str(r.get("decision", "")).upper() == "YES":
                yes += 1

        if stopped:
            break

    score = (yes / total * 100) if total else 0.0
    low, high = _wilson_ci_95(yes, total)
    return {
        "mode": mode,
        "judge_policy": JUDGE_POLICY,
        "rubrics_path": path,
        "total_gold_atomic_points": total,
        "count_yes": yes,
        "cia_score_percent": score,
        "ci_95_percent": {
            "low": (low * 100 if low is not None else None),
            "high": (high * 100 if high is not None else None),
        },
        "processed_new_items": processed_new,
        "cache_hit_items": cache_hit,
        "skipped_missing_generated": skipped_missing,
        "error_messages": error_messages,
        "stopped": stopped,
        "message": stop_message,
        "started_at": started,
        "finished_at": _utc_now(),
        "elapsed_seconds": round(datetime.now(timezone.utc).timestamp() - t0, 3),
    }


def main():
    if not API_KEY:
        print("❌ Missing API key.")
        return
    if not GOLD_ATOMIC_PATH.exists():
        print(f"❌ Gold atomic file not found: {GOLD_ATOMIC_PATH}")
        return
    gold = load_gold_atomic(GOLD_ATOMIC_PATH)
    if not gold:
        print("❌ Gold atomic points empty.")
        return

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60.0)

    summary = {
        "script": "LLMEval_Med/eval_cia_batch_zh.py",
        "started_at": _utc_now(),
        "finished_at": None,
        "judge_model": JUDGE_MODEL,
        "judge_policy": JUDGE_POLICY,
        "api_provider": API_PROVIDER,
        "gold_atomic_path": str(GOLD_ATOMIC_PATH),
        "cache_root": str(CACHE_ROOT),
        "summary_path": str(SUMMARY_PATH),
        "range": {"start": START_INDEX, "end": END_INDEX},
        "modes_requested": MODES,
        "p_value_baseline_mode": P_VALUE_BASELINE_MODE,
        "modes": [],
    }

    for mode in MODES:
        print(f"\n🔬 CIA mode={mode} range={START_INDEX}-{END_INDEX} policy={JUDGE_POLICY}")
        stats = run_mode(client, mode, gold)
        if stats is None:
            continue
        summary["modes"].append(stats)
        if stats.get("stopped"):
            break

    baseline = None
    for m in summary["modes"]:
        if m.get("mode") == P_VALUE_BASELINE_MODE and not m.get("stopped"):
            baseline = m
            break
    for m in summary["modes"]:
        if m.get("stopped"):
            m[f"p_value_vs_{P_VALUE_BASELINE_MODE}"] = None
            continue
        if baseline is None or m.get("mode") == P_VALUE_BASELINE_MODE:
            m[f"p_value_vs_{P_VALUE_BASELINE_MODE}"] = None
            continue
        pv = _two_proportion_pvalue(
            m.get("count_yes", 0),
            m.get("total_gold_atomic_points", 0),
            baseline.get("count_yes", 0),
            baseline.get("total_gold_atomic_points", 0),
        )
        m[f"p_value_vs_{P_VALUE_BASELINE_MODE}"] = pv

    summary["finished_at"] = _utc_now()
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ CIA summary saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

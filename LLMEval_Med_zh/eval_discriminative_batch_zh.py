import ast
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI


START_INDEX = int(os.getenv("START_INDEX", 0))
END_INDEX = int(os.getenv("END_INDEX", 10**9))
N_REPEAT = max(1, int(os.getenv("N_REPEAT", 3)))
DO_ORDER_SWAP = str(os.getenv("DO_ORDER_SWAP", "1")).strip().lower() in ("1", "true", "yes", "y", "on")
SAVE_RAW = str(os.getenv("DISCRIM_SAVE_RAW", "0")).strip().lower() in ("1", "true", "yes", "y", "on")

BASE_DIR = Path(__file__).resolve().parent
PAIRWISE_DATASET = Path(
    os.getenv(
        "PAIRWISE_DATASET",
        str(BASE_DIR / "outputs" / "eval" / "pairwise_dataset_perturbed_zh.jsonl"),
    )
)
OUTPUT_DIR = Path(
    os.getenv(
        "DISCRIM_OUTPUT_DIR",
        str(BASE_DIR / "outputs" / "eval" / "discrimination"),
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODE_RUBRIC_PATHS = {
    "ours": str(BASE_DIR / "outputs" / "rubrics_generated_zh.jsonl"),
    "none": None,
    "generic": str(BASE_DIR / "outputs" / "eval" / "generic_rubrics.jsonl"),
}
# Extend modes by env: DISCRIM_EXTRA_MODE_PATHS="generic=data/generic_rubrics.jsonl,gpt4o=data/rubrics_GPT4o_converted.jsonl"
extra = os.getenv("DISCRIM_EXTRA_MODE_PATHS", "").strip()
if extra:
    for pair in extra.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k:
            MODE_RUBRIC_PATHS[k] = v or None

raw_modes = os.getenv("DISCRIM_MODES", "ours,none,generic")
MODES = [m.strip() for m in raw_modes.split(",") if m.strip() in MODE_RUBRIC_PATHS]
if not MODES:
    MODES = ["ours", "none"]

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
TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", 0.0))
TIMEOUT = float(os.getenv("JUDGE_TIMEOUT", 60.0))

RUN_SUMMARY_PATH = Path(
    os.getenv(
        "DISCRIM_RUN_SUMMARY_PATH",
        str(
            OUTPUT_DIR
            / f"run_summary_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        ),
    )
)

QUOTA_ERROR_KEYWORDS = (
    "insufficient",
    "quota",
    "rate limit",
    "429",
    "billing",
    "credit",
    "exceeded your current quota",
    "tokens per day limit exceeded",
    "token",
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
    import re
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _load_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _extract_rubrics(raw) -> List[Dict]:
    if isinstance(raw, dict):
        items = raw.get("rubrics", [])
    elif isinstance(raw, list):
        items = raw
    else:
        items = []
    out = []
    for x in items:
        if not isinstance(x, dict):
            continue
        c = str(x.get("criterion", "")).strip()
        if not c:
            continue
        out.append(
            {
                "criterion": c,
                "axis": str(x.get("axis", "overall")).strip() or "overall",
                "points": float(x.get("points", 0)),
            }
        )
    return out


def _load_generic_rubrics(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8").strip()
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
        return _extract_rubrics(parsed.get("rubrics", []))
    if isinstance(parsed, list):
        return _extract_rubrics(parsed)

    out: List[Dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        out.extend(_extract_rubrics([obj]))
    return out


def _load_mode_rubrics(mode: str) -> Tuple[str, Optional[Dict[int, List[Dict]]], Optional[List[Dict]]]:
    path = MODE_RUBRIC_PATHS.get(mode)
    if mode == "none" or path is None:
        return "none", None, None
    p = Path(path)
    if not p.exists():
        return "missing", None, None
    if mode == "generic" or p.name == "generic_rubrics.jsonl":
        g = _load_generic_rubrics(path)
        return "generic", None, g

    rubrics_by_idx: Dict[int, List[Dict]] = {}
    for row in _load_jsonl(p):
        try:
            idx = int(row.get("index", row.get("conv_id")))
        except Exception:
            continue
        raw = row.get("generated_rubrics", row.get("rubrics", []))
        rubrics = _extract_rubrics(raw)
        if rubrics:
            rubrics_by_idx[idx] = rubrics
    return "indexed", rubrics_by_idx, None


def _format_rubrics(items: List[Dict]) -> str:
    lines = [
        "Apply each rule independently. Add points when behavior is present; subtract points when undesired behavior appears."
    ]
    for i, it in enumerate(items, 1):
        sign = "+" if float(it.get("points", 0)) >= 0 else ""
        lines.append(
            f"{i}. ({sign}{float(it.get('points', 0))} points, axis={it.get('axis','overall')}) {it.get('criterion','')}"
        )
    return "\n".join(lines)


def _build_prompt(rubric_text: str, question: str, a: str, b: str) -> str:
    if rubric_text.strip():
        header = (
            "Compare Response A vs Response B using the provided rubric.\n\n"
            f"[RUBRIC]\n{rubric_text}"
        )
        rule = "items[].id must match rubric numbering (1..N)."
    else:
        header = (
            "Compare Response A vs Response B based on overall clinical accuracy, safety, and communication quality."
        )
        rule = "Provide one overall item in items."
    return f"""
You are a strict clinical evaluator. {header}

[QUESTION]
{question}

[RESPONSE A]
{a}

[RESPONSE B]
{b}

Return JSON ONLY:
{{
  "decision": "A|B|SAME",
  "total": {{"A": <number>, "B": <number>, "delta": <number>}},
  "items": [
    {{"id": <int>, "axis": "<string>", "points": <number>, "hit_A": <true/false>, "hit_B": <true/false>}}
  ]
}}

Rules:
- {rule}
- Sum item points to compute totals.
- decision: A if delta>0, B if delta<0, SAME if close.
- No extra text.
""".strip()


def _judge_once(client: OpenAI, rubric_text: str, question: str, a: str, b: str):
    prompt = _build_prompt(rubric_text, question, a, b)
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )
    raw = (resp.choices[0].message.content or "").strip()
    parsed = _safe_parse_json(raw)
    return parsed, raw


def _write_summary(summary: Dict):
    RUN_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def run_mode(client: OpenAI, mode: str) -> Dict:
    output_file = OUTPUT_DIR / f"results_{mode}.jsonl"
    started = _utc_now()
    t0 = datetime.now(timezone.utc).timestamp()

    rubric_type, rubrics_map, generic_items = _load_mode_rubrics(mode)
    if rubric_type == "missing":
        return {
            "mode": mode,
            "status": "skipped",
            "reason": "rubric_file_missing",
            "started_at": started,
            "finished_at": _utc_now(),
            "elapsed_seconds": 0.0,
        }
    if rubric_type == "generic" and not generic_items:
        return {
            "mode": mode,
            "status": "skipped",
            "reason": "generic_rubrics_empty",
            "started_at": started,
            "finished_at": _utc_now(),
            "elapsed_seconds": 0.0,
        }

    expected_n = (2 if DO_ORDER_SWAP else 1) * N_REPEAT
    done_pairs = set()
    kept_lines: List[str] = []
    stale = 0
    if output_file.exists():
        for line in output_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                stale += 1
                continue
            pid = str(obj.get("pair_id", ""))
            results = obj.get("results", [])
            if pid and isinstance(results, list) and len(results) == expected_n:
                done_pairs.add(pid)
                kept_lines.append(json.dumps(obj, ensure_ascii=False))
            else:
                stale += 1
    if stale > 0:
        output_file.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")

    processed = 0
    skipped_range = 0
    skipped_done = 0
    skipped_no_rubric = 0
    skipped_eval_error = 0
    error_messages: List[str] = []

    rows = _load_jsonl(PAIRWISE_DATASET)
    with output_file.open("a", encoding="utf-8") as fout:
        for ex in rows:
            try:
                idx = int(ex.get("conv_id", -1))
            except Exception:
                continue
            pair_id = str(ex.get("pair_id", "")).strip()
            if not pair_id:
                continue
            if not (START_INDEX <= idx <= END_INDEX):
                skipped_range += 1
                continue
            if pair_id in done_pairs:
                skipped_done += 1
                continue

            rubric_text = ""
            if rubric_type == "indexed":
                if idx not in (rubrics_map or {}):
                    skipped_no_rubric += 1
                    continue
                rubric_text = _format_rubrics((rubrics_map or {})[idx])
            elif rubric_type == "generic":
                rubric_text = _format_rubrics(generic_items or [])

            question = str(ex.get("question", "")).strip()
            a_text = str((ex.get("A") or {}).get("text", "")).strip()
            b_text = str((ex.get("B") or {}).get("text", "")).strip()
            if not question or not a_text or not b_text:
                skipped_eval_error += 1
                continue

            orders = [(a_text, b_text)]
            if DO_ORDER_SWAP:
                orders.append((b_text, a_text))

            results = []
            pair_failed = False
            for a_cur, b_cur in orders:
                for _ in range(N_REPEAT):
                    try:
                        parsed, raw = _judge_once(client, rubric_text, question, a_cur, b_cur)
                        rec = {"parsed": parsed, "A_text": a_cur == a_text}
                        if SAVE_RAW:
                            rec["raw"] = raw
                        results.append(rec)
                    except Exception as e:
                        msg = _extract_error_message(e)
                        if msg and len(error_messages) < 50:
                            error_messages.append(msg)
                        if _is_quota_error_text(msg):
                            return {
                                "mode": mode,
                                "status": "stopped_quota",
                                "message": msg,
                                "processed_new_pairs": processed,
                                "skipped_out_of_range": skipped_range,
                                "skipped_done_pairs": skipped_done,
                                "skipped_missing_rubric": skipped_no_rubric,
                                "skipped_pair_eval_error": skipped_eval_error,
                                "error_messages": error_messages,
                                "started_at": started,
                                "finished_at": _utc_now(),
                                "elapsed_seconds": round(datetime.now(timezone.utc).timestamp() - t0, 3),
                            }
                        pair_failed = True
                        break
                if pair_failed:
                    break

            if pair_failed or len(results) != expected_n:
                skipped_eval_error += 1
                continue

            fout.write(
                json.dumps(
                    {
                        "pair_id": pair_id,
                        "conv_id": idx,
                        "mode": mode,
                        "results": results,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fout.flush()
            processed += 1

    return {
        "mode": mode,
        "status": "finished",
        "processed_new_pairs": processed,
        "skipped_out_of_range": skipped_range,
        "skipped_done_pairs": skipped_done,
        "skipped_missing_rubric": skipped_no_rubric,
        "skipped_pair_eval_error": skipped_eval_error,
        "error_messages": error_messages,
        "started_at": started,
        "finished_at": _utc_now(),
        "elapsed_seconds": round(datetime.now(timezone.utc).timestamp() - t0, 3),
    }


def main():
    if not API_KEY:
        print("❌ Missing API key.")
        return
    if not PAIRWISE_DATASET.exists():
        print(f"❌ Pairwise dataset not found: {PAIRWISE_DATASET}")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=TIMEOUT)
    summary = {
        "script": "LLMEval_Med/eval_discriminative_batch_zh.py",
        "started_at": _utc_now(),
        "finished_at": None,
        "config": {
            "judge_model": JUDGE_MODEL,
            "api_provider": API_PROVIDER,
            "pairwise_dataset": str(PAIRWISE_DATASET),
            "output_dir": str(OUTPUT_DIR),
            "start_index": START_INDEX,
            "end_index": END_INDEX,
            "n_repeat": N_REPEAT,
            "do_order_swap": DO_ORDER_SWAP,
            "save_raw": SAVE_RAW,
            "modes": MODES,
            "mode_rubric_paths": MODE_RUBRIC_PATHS,
        },
        "modes": [],
    }
    _write_summary(summary)

    for mode in MODES:
        print(f"\n🚀 Discrimination mode={mode} range={START_INDEX}-{END_INDEX}")
        stats = run_mode(client, mode)
        summary["modes"].append(stats)
        _write_summary(summary)
        if stats.get("status") == "stopped_quota":
            break

    summary["finished_at"] = _utc_now()
    _write_summary(summary)
    print(f"\n✅ Run summary saved: {RUN_SUMMARY_PATH}")


if __name__ == "__main__":
    main()

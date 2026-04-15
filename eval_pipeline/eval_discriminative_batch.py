import ast
import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from config import API_PROVIDER, DO_ORDER_SWAP, JUDGE_MODEL, N_REPEAT
from pairwise_judge import judge_once
from rubric_utils import format_rubrics, load_rubrics_by_index


# ================= CONFIG =================
START_INDEX = int(os.getenv("START_INDEX", 0))
END_INDEX = int(os.getenv("END_INDEX", 300))

DATA_DIR = "data_evaluation"
PAIRWISE_DATASET = os.getenv("PAIRWISE_DATASET", "data/pairwise_dataset_perturbed.jsonl")
OUTPUT_DIR = os.getenv("DISCRIM_OUTPUT_DIR", "outputs/ablation_discrim")

MODE_RUBRIC_PATHS = {
    "full": os.path.join(DATA_DIR, "rubrics_full.jsonl"),
    "no_router": os.path.join(DATA_DIR, "rubrics_no_router.jsonl"),
    "no_atomic": os.path.join(DATA_DIR, "rubrics_no_atomic.jsonl"),
    "no_intent": os.path.join(DATA_DIR, "rubrics_no_intent.jsonl"),
    "no_audit": os.path.join(DATA_DIR, "rubrics_no_audit.jsonl"),
    "gpt4o": "data/rubrics_GPT4o_converted.jsonl",
    "generic": "data/generic_rubrics.jsonl",
    "none": None,
}

DEFAULT_MODES = ["full", "no_router", "no_atomic", "no_intent", "no_audit"]

QUOTA_ERROR_KEYWORDS = (
    "quota",
    "insufficient",
    "rate limit",
    "429",
    "billing",
    "credit",
    "exceeded your current quota",
    "token",
    "quota_exhausted",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_modes() -> List[str]:
    raw = os.getenv("DISCRIM_MODES", "").strip()
    if not raw:
        return DEFAULT_MODES

    out: List[str] = []
    for part in raw.split(","):
        mode = part.strip()
        if not mode:
            continue
        if mode not in MODE_RUBRIC_PATHS:
            print(f"⚠️ Unknown mode [{mode}] in DISCRIM_MODES, ignoring.")
            continue
        if mode not in out:
            out.append(mode)
    return out or DEFAULT_MODES


MODES = _parse_modes()
os.makedirs(OUTPUT_DIR, exist_ok=True)
RUN_SUMMARY_PATH = os.getenv(
    "DISCRIM_RUN_SUMMARY_PATH",
    os.path.join(
        OUTPUT_DIR,
        f"run_summary_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",
    ),
)


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


EVAL_N_REPEAT = int(os.getenv("N_REPEAT", str(N_REPEAT)))
EVAL_N_REPEAT = max(1, EVAL_N_REPEAT)
EVAL_DO_ORDER_SWAP = _parse_bool_env("DO_ORDER_SWAP", bool(DO_ORDER_SWAP))
DISCRIM_SAVE_RAW = _parse_bool_env("DISCRIM_SAVE_RAW", False)


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
        criterion = str(x.get("criterion", "")).strip()
        if not criterion:
            continue
        out.append(
            {
                "criterion": criterion,
                "axis": str(x.get("axis", "overall")),
                "points": x.get("points", 0),
            }
        )
    return out


def _load_generic_rubrics(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []

    text = open(path, "r", encoding="utf-8").read().strip()
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

    if parsed is not None:
        return _extract_rubrics(parsed)

    items: List[Dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            candidate = _extract_rubrics([obj])
            if candidate:
                items.extend(candidate)
    return items


def _load_mode_rubrics(mode: str) -> Tuple[str, Optional[Dict[int, List[Dict]]], Optional[List[Dict]]]:
    if mode == "none":
        return "none", None, None
    if mode == "generic":
        items = _load_generic_rubrics(MODE_RUBRIC_PATHS["generic"])
        return "generic", None, items

    path = MODE_RUBRIC_PATHS.get(mode)
    if not path or not os.path.exists(path):
        return "missing", None, None
    return "indexed", load_rubrics_by_index(path), None


def _write_summary(summary: Dict):
    with open(RUN_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


async def run_evaluation(mode: str):
    output_file = os.path.join(OUTPUT_DIR, f"results_{mode}.jsonl")
    mode_started = _utc_now()
    mode_start_ts = datetime.now(timezone.utc).timestamp()

    rubric_type, rubrics_map, generic_items = _load_mode_rubrics(mode)
    if rubric_type == "missing":
        msg = f"Rubric file missing for mode [{mode}], skipping."
        print(f"⚠️  {msg}")
        return {
            "mode": mode,
            "status": "skipped",
            "reason": msg,
            "started_at": mode_started,
            "finished_at": _utc_now(),
            "elapsed_seconds": 0.0,
        }
    if rubric_type == "generic" and not generic_items:
        msg = f"Generic rubrics empty for mode [{mode}], skipping."
        print(f"⚠️  {msg}")
        return {
            "mode": mode,
            "status": "skipped",
            "reason": msg,
            "started_at": mode_started,
            "finished_at": _utc_now(),
            "elapsed_seconds": 0.0,
        }

    print(f"\n🚀 Running Discriminative Eval for Mode: [{mode}] (Range {START_INDEX}-{END_INDEX})")
    if rubric_type == "indexed":
        print(f"   Loaded {len(rubrics_map or {})} indexed rubrics.")
    elif rubric_type == "generic":
        print(f"   Loaded {len(generic_items or [])} generic rubrics.")
    else:
        print("   Running no-rubric baseline (mode=none).")

    done_pairs = set()
    expected_results_per_pair = (2 if EVAL_DO_ORDER_SWAP else 1) * EVAL_N_REPEAT
    stale_records = 0
    kept_lines: List[str] = []
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    pair_id = str(obj.get("pair_id", ""))
                    results = obj.get("results", [])
                    has_swap = False
                    if isinstance(results, list):
                        flags = [r.get("A_text") for r in results if isinstance(r, dict)]
                        has_swap = (True in flags) and (False in flags)
                    is_valid = (
                        bool(pair_id)
                        and isinstance(results, list)
                        and len(results) == expected_results_per_pair
                        and ((not EVAL_DO_ORDER_SWAP) or has_swap)
                    )
                    if is_valid:
                        done_pairs.add(pair_id)
                        kept_lines.append(line if line.endswith("\n") else line + "\n")
                    else:
                        stale_records += 1
                except Exception:
                    stale_records += 1
        if stale_records > 0:
            with open(output_file, "w", encoding="utf-8") as fw:
                fw.writelines(kept_lines)
            print(
                f"   ♻️ Removed {stale_records} stale/incompatible records "
                f"(expect results={expected_results_per_pair}, swap={EVAL_DO_ORDER_SWAP})."
            )
    print(f"   ⏭️  Skipping {len(done_pairs)} already processed pairs.")

    processed = 0
    skipped_range = 0
    skipped_done = 0
    skipped_no_rubric = 0
    skipped_pair_eval_error = 0
    errors = 0
    error_messages: List[str] = []

    with open(PAIRWISE_DATASET, "r", encoding="utf-8") as fin, open(
        output_file, "a", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            ex = json.loads(line)

            idx = int(ex.get("conv_id", -1))
            pair_id = str(ex.get("pair_id", ""))

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
                rubric_text = format_rubrics((rubrics_map or {})[idx])
            elif rubric_type == "generic":
                rubric_text = format_rubrics(generic_items or [])

            question = str(ex.get("question", ""))
            A = str((ex.get("A") or {}).get("text", ""))
            B = str((ex.get("B") or {}).get("text", ""))

            orders = [(A, B)]
            if EVAL_DO_ORDER_SWAP:
                orders.append((B, A))

            results = []
            pair_failed = False
            for a_text, b_text in orders:
                for _ in range(EVAL_N_REPEAT):
                    try:
                        parsed, raw = judge_once(rubric_text, question, a_text, b_text)
                        rec = {
                            "parsed": parsed,
                            "A_text": a_text == A,
                        }
                        if DISCRIM_SAVE_RAW:
                            rec["raw"] = raw
                        results.append(rec)
                    except Exception as e:
                        err_msg = _extract_error_message(e)
                        if _is_quota_error_text(err_msg):
                            print(f"\n   🛑 API quota exhausted while judging pair {pair_id} (Conv {idx}).")
                            print(f"   🛑 message: {err_msg}")
                            print("   🛑 Stopping discriminative batch immediately.")
                            return {
                                "mode": mode,
                                "status": "stopped_quota",
                                "message": err_msg,
                                "processed_new_pairs": processed,
                                "skipped_out_of_range": skipped_range,
                                "skipped_done_pairs": skipped_done,
                                "skipped_missing_rubric": skipped_no_rubric,
                                "skipped_pair_eval_error": skipped_pair_eval_error,
                                "errors": errors,
                                "error_messages": error_messages[:50],
                                "started_at": mode_started,
                                "finished_at": _utc_now(),
                                "elapsed_seconds": round(
                                    datetime.now(timezone.utc).timestamp() - mode_start_ts, 3
                                ),
                            }
                        errors += 1
                        if err_msg and len(error_messages) < 50:
                            error_messages.append(err_msg)
                        print(f"   ❌ Error judging pair {pair_id}: {err_msg}")
                        pair_failed = True
                        break
                if pair_failed:
                    break

            if pair_failed or len(results) != expected_results_per_pair:
                skipped_pair_eval_error += 1
                continue

            out_record = {
                "pair_id": pair_id,
                "conv_id": idx,
                "mode": mode,
                "results": results,
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            fout.flush()
            processed += 1
            print(f"   [Mode: {mode}] Processed Pair {pair_id} (Conv {idx})", end="\r")

    print(f"\n   ✅ Mode {mode} finished. Processed {processed} new pairs.")
    return {
        "mode": mode,
        "status": "finished",
        "processed_new_pairs": processed,
        "skipped_out_of_range": skipped_range,
        "skipped_done_pairs": skipped_done,
        "skipped_missing_rubric": skipped_no_rubric,
        "skipped_pair_eval_error": skipped_pair_eval_error,
        "errors": errors,
        "error_messages": error_messages[:50],
        "started_at": mode_started,
        "finished_at": _utc_now(),
        "elapsed_seconds": round(datetime.now(timezone.utc).timestamp() - mode_start_ts, 3),
    }


async def main():
    started_at = _utc_now()
    summary = {
        "script": "eval_pipeline/eval_discriminative_batch.py",
        "started_at": started_at,
        "finished_at": None,
        "config": {
            "judge_model": JUDGE_MODEL,
            "api_provider": API_PROVIDER,
            "pairwise_dataset": PAIRWISE_DATASET,
            "output_dir": OUTPUT_DIR,
            "start_index": START_INDEX,
            "end_index": END_INDEX,
            "n_repeat": EVAL_N_REPEAT,
            "do_order_swap": EVAL_DO_ORDER_SWAP,
            "save_raw": DISCRIM_SAVE_RAW,
            "modes": MODES,
        },
        "modes": [],
    }
    _write_summary(summary)

    for mode in MODES:
        mode_stats = await run_evaluation(mode)
        summary["modes"].append(mode_stats)
        _write_summary(summary)
        if mode_stats.get("status") == "stopped_quota":
            break

    summary["finished_at"] = _utc_now()
    _write_summary(summary)
    print(f"\n📝 Run summary saved to: {RUN_SUMMARY_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

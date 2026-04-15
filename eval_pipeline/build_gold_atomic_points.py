import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from config import BASE_URL, OPENAI_API_KEY, JUDGE_MODEL, TEMPERATURE


# ================= CONFIG =================
SOURCE_CANDIDATES = [
    "data/gold_rubrics.jsonl",
    "data/reference_rubrics_all.jsonl",
    #"data/final_rubrics_refined_15.jsonl",
]
OUTPUT_PATH = "data_evaluation/gold_atomic_points_all.jsonl"
CACHE_DIR = "outputs/gold_atomic_cache"

START_INDEX = int(os.getenv("START_INDEX", 0))
END_INDEX = int(os.getenv("END_INDEX", 300))
MAX_RETRY = int(os.getenv("ATOMIZE_MAX_RETRY", 3))
RETRY_SLEEP = float(os.getenv("ATOMIZE_RETRY_SLEEP", 2.0))
ATOMIZER_MODEL = os.getenv("ATOMIZER_MODEL", JUDGE_MODEL)
REBUILD_ONLY_FROM_CACHE = (
    str(os.getenv("REBUILD_ONLY_FROM_CACHE", "")).strip().lower()
    in ("1", "true", "yes", "y")
)

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
)


client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL) if OpenAI else None


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

    match = re.search(r"\{[\s\S]*\}", text or "")
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _select_source_file() -> Optional[str]:
    for path in SOURCE_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def _count_existing_cache_ids(cache_dir: str, start_idx: int, end_idx: int) -> int:
    if not os.path.exists(cache_dir):
        return 0
    count = 0
    for name in os.listdir(cache_dir):
        if not name.endswith(".json"):
            continue
        stem = name[:-5]
        try:
            cid = int(stem)
        except Exception:
            continue
        if start_idx <= cid <= end_idx:
            count += 1
    return count


def _normalize_rubrics(raw_rubrics) -> List[Dict]:
    if isinstance(raw_rubrics, dict):
        rubrics = raw_rubrics.get("rubrics", [])
    elif isinstance(raw_rubrics, list):
        rubrics = raw_rubrics
    else:
        rubrics = []

    normalized = []
    for item in rubrics:
        if not isinstance(item, dict):
            continue
        criterion = str(item.get("criterion", "")).strip()
        if not criterion:
            continue
        normalized.append(
            {
                "criterion": criterion,
                "axis": str(item.get("axis", "unknown")).strip() or "unknown",
                "points": item.get("points", None),
            }
        )
    return normalized


def _infer_polarity(criterion: str, points) -> str:
    text = (criterion or "").lower()
    neg_markers = ("do not", "don't", "avoid", "fails to", "must not", "never", "omit")
    if isinstance(points, (int, float)) and points < 0:
        return "negative_constraint"
    if any(m in text for m in neg_markers):
        return "negative_constraint"
    return "positive_requirement"


def _build_fallback_atomic_points(rubrics: List[Dict]) -> List[Dict]:
    atomic_points = []
    for idx, r in enumerate(rubrics, start=1):
        atomic_points.append(
            {
                "text": r["criterion"],
                "source_criterion_index": idx,
                "axis": r.get("axis", "unknown"),
                "polarity": _infer_polarity(r["criterion"], r.get("points")),
            }
        )
    return atomic_points


def _atomize_via_llm(conv_id: int, rubrics: List[Dict]) -> Tuple[List[Dict], str]:
    if client is None:
        raise RuntimeError("openai package is not installed")
    rubric_lines = []
    for i, r in enumerate(rubrics, start=1):
        rubric_lines.append(
            f"{i}. [axis={r.get('axis','unknown')}, points={r.get('points')}] {r['criterion']}"
        )
    rubric_block = "\n".join(rubric_lines)

    prompt = f"""
You are converting physician-authored reference rubrics into atomic keypoints.

Atomicity definition:
1) Indivisible: each point has exactly one checkable action, medical fact, or interaction requirement.
2) Self-contained: Must be fully understandable in isolation. You must RESOLVE all pronouns (e.g., replace "it", "they", "their", "those", "above", "the patient", "the drug") with specific nouns based on context.
3) Binary-verifiable: evaluator can answer YES/NO without ambiguity.

Decomposition rules:
1) Split compound instructions into separate atomic points.
2) Preserve conditional logic as one unit (do NOT split condition from action).
3) Keep explicit negative constraints/prohibitions as independent atomic points.

Input Rubrics (Conversation ID: {conv_id}):
{rubric_block}

Return strict JSON only:
{{
  "gold_atomic_points": [
    {{
      "text": "atomic point text",
      "source_criterion_index": 1,
      "axis": "accuracy|completeness|context_awareness|communication_quality|instruction_following|other",
      "polarity": "positive_requirement|negative_constraint"
    }}
  ]
}}
"""
    last_error = ""
    for attempt in range(MAX_RETRY):
        try:
            resp = client.chat.completions.create(
                model=ATOMIZER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
            )
            raw = (resp.choices[0].message.content or "").strip()
            parsed = _safe_parse_json(raw)
            if isinstance(parsed, dict) and isinstance(parsed.get("gold_atomic_points"), list):
                return parsed["gold_atomic_points"], raw
            last_error = "invalid_json_output"
        except Exception as e:
            last_error = _extract_error_message(e)
            print(f"[Atomize Retry {attempt+1}] ID {conv_id} error: {last_error}")
            if _is_quota_error_text(last_error):
                raise RuntimeError(last_error) from e
            time.sleep(RETRY_SLEEP + attempt)

    raise RuntimeError(f"Atomization failed for conv_id={conv_id}: {last_error}")


def _sanitize_atomic_points(raw_points: List[Dict], rubrics: List[Dict]) -> List[Dict]:
    sanitized = []
    seen = set()
    n = len(rubrics)

    for p in raw_points:
        if not isinstance(p, dict):
            continue
        text = str(p.get("text", "")).strip()
        if not text:
            continue

        src = p.get("source_criterion_index", 1)
        try:
            src = int(src)
        except Exception:
            src = 1
        if src < 1 or src > max(1, n):
            src = 1

        axis = str(p.get("axis", "")).strip() or rubrics[src - 1].get("axis", "unknown")
        polarity = str(p.get("polarity", "")).strip()
        if polarity not in ("positive_requirement", "negative_constraint"):
            polarity = _infer_polarity(text, rubrics[src - 1].get("points"))

        key = re.sub(r"\s+", " ", text.lower())
        if key in seen:
            continue
        seen.add(key)

        sanitized.append(
            {
                "text": text,
                "source_criterion_index": src,
                "axis": axis,
                "polarity": polarity,
            }
        )

    return sanitized


def _rebuild_output_from_cache(source_path: str) -> int:
    written = 0
    with open(source_path, "r", encoding="utf-8") as fin, open(
        OUTPUT_PATH, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            raw_id = obj.get("index", obj.get("conv_id"))
            if raw_id is None:
                continue
            try:
                conv_id = int(raw_id)
            except Exception:
                continue

            if not (START_INDEX <= conv_id <= END_INDEX):
                continue

            rubrics = _normalize_rubrics(obj.get("rubrics", obj.get("generated_rubrics", [])))
            if not rubrics:
                continue

            cache_file = os.path.join(CACHE_DIR, f"{conv_id}.json")
            if not os.path.exists(cache_file):
                continue

            try:
                with open(cache_file, "r", encoding="utf-8") as cf:
                    cached = json.load(cf)
                atomic_points = cached.get("gold_atomic_points", [])
            except Exception:
                atomic_points = []
            if not atomic_points:
                continue

            enriched = []
            for i, p in enumerate(atomic_points, start=1):
                src_idx = int(p.get("source_criterion_index", 1))
                if src_idx < 1 or src_idx > len(rubrics):
                    src_idx = 1
                source_criterion = rubrics[src_idx - 1]["criterion"] if rubrics else ""
                enriched.append(
                    {
                        "ga_id": f"{conv_id}_{i}",
                        "text": p.get("text", ""),
                        "axis": p.get("axis", "unknown"),
                        "polarity": p.get("polarity", "positive_requirement"),
                        "source_criterion_index": src_idx,
                        "source_criterion": source_criterion,
                    }
                )

            out_record = {
                "conv_id": str(conv_id),
                "source_rubric_count": len(rubrics),
                "gold_atomic_count": len(enriched),
                "gold_atomic_points": enriched,
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            written += 1
    return written


def main():
    source_path = _select_source_file()
    if not source_path:
        print("❌ No reference rubrics source found.")
        print(f"   Tried: {SOURCE_CANDIDATES}")
        return

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    existing_cache_count = _count_existing_cache_ids(CACHE_DIR, START_INDEX, END_INDEX)
    print(f"🚀 Building gold atomic points from: {source_path}")
    print(f"📦 Existing cache records: {existing_cache_count}")
    print(f"🔢 Range: {START_INDEX} - {END_INDEX}")

    if REBUILD_ONLY_FROM_CACHE:
        written = _rebuild_output_from_cache(source_path)
        print(f"🧱 Rebuilt output from cache only: {OUTPUT_PATH} ({written} records)")
        return

    if client is None:
        print("❌ openai package is not installed. Cannot generate missing cache entries.")
        print("   Tip: run with REBUILD_ONLY_FROM_CACHE=1 to rebuild output from existing cache only.")
        return

    new_count = 0
    stopped_early = False
    with open(source_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            raw_id = obj.get("index", obj.get("conv_id"))
            if raw_id is None:
                continue
            try:
                conv_id = int(raw_id)
            except Exception:
                continue

            if not (START_INDEX <= conv_id <= END_INDEX):
                continue

            rubrics = _normalize_rubrics(obj.get("rubrics", obj.get("generated_rubrics", [])))
            if not rubrics:
                continue

            cache_file = os.path.join(CACHE_DIR, f"{conv_id}.json")
            atomic_points = []
            raw_output = ""
            cache_valid = False

            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as cf:
                        cached = json.load(cf)
                    atomic_points = cached.get("gold_atomic_points", [])
                    raw_output = cached.get("raw_output", "")
                    cache_valid = isinstance(atomic_points, list) and len(atomic_points) > 0
                except Exception:
                    atomic_points = []
                    cache_valid = False

            # 断点续跑：只按 cache 是否存在且有效来判断
            if cache_valid:
                continue

            try:
                llm_points, raw_output = _atomize_via_llm(conv_id, rubrics)
                atomic_points = _sanitize_atomic_points(llm_points, rubrics)
            except Exception as e:
                msg = _extract_error_message(e)
                print(f"🛑 Atomization error at conv_id={conv_id}: {msg}")
                if _is_quota_error_text(msg):
                    print("🛑 API quota/token issue detected. Stopping now.")
                    stopped_early = True
                    break
                atomic_points = _build_fallback_atomic_points(rubrics)
                raw_output = f"fallback_due_to_error: {msg}"

            if not atomic_points:
                atomic_points = _build_fallback_atomic_points(rubrics)

            with open(cache_file, "w", encoding="utf-8") as cf:
                json.dump(
                    {
                        "conv_id": conv_id,
                        "source_rubric_count": len(rubrics),
                        "gold_atomic_count": len(atomic_points),
                        "gold_atomic_points": atomic_points,
                        "raw_output": raw_output,
                    },
                    cf,
                    indent=2,
                    ensure_ascii=False,
                )

            new_count += 1
            print(
                f"   ✅ conv_id={conv_id} atomic_points={len(atomic_points)} (new={new_count})",
                end="\r",
            )

    written = _rebuild_output_from_cache(source_path)
    print(f"\n🧱 Rebuilt output from cache: {OUTPUT_PATH} ({written} records)")
    if stopped_early:
        print(f"🛑 Stopped early. Newly generated cache records: {new_count}")
        return
    print(f"🎉 Done. Newly generated cache records: {new_count}")


if __name__ == "__main__":
    main()

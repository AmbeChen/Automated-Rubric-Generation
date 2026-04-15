import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


START_INDEX = int(os.getenv("START_INDEX", 0))
END_INDEX = int(os.getenv("END_INDEX", 10**9))
MAX_RETRY = int(os.getenv("ATOMIZE_MAX_RETRY", 3))
RETRY_SLEEP = float(os.getenv("ATOMIZE_RETRY_SLEEP", 2.0))

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = Path(os.getenv("GOLD_RUBRICS_PATH", str(BASE_DIR / "outputs" / "eval" / "gold_reference_rubrics_zh.jsonl")))
OUTPUT_PATH = Path(os.getenv("GOLD_ATOMIC_OUTPUT_PATH", str(BASE_DIR / "outputs" / "eval" / "gold_atomic_points_zh.jsonl")))
CACHE_DIR = Path(os.getenv("GOLD_ATOMIC_CACHE_DIR", str(BASE_DIR / "outputs" / "eval" / "gold_atomic_cache")))

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

ATOMIZER_MODEL = os.getenv("ATOMIZER_MODEL", os.getenv("JUDGE_MODEL", "gpt-4.1-mini"))
TEMPERATURE = float(os.getenv("ATOMIZE_TEMPERATURE", "0.0"))
# 非中文原子点处理策略：
# - source_fallback: 直接回退到对应 source criterion（默认，无额外 API 开销）
# - keep: 保持原样
NON_CJK_STRATEGY = os.getenv("ATOMIZE_NON_CJK_STRATEGY", "source_fallback").strip().lower()

QUOTA_ERROR_KEYWORDS = (
    "error code",
    "Rate limit reached for model",
    "exceeded your current quota",
    "tokens per day limit exceeded",
)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _has_cjk(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(CJK_RE.search(text))


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
    for r in items:
        if not isinstance(r, dict):
            continue
        c = str(r.get("criterion", "")).strip()
        if not c:
            continue
        out.append(
            {
                "criterion": c,
                "axis": str(r.get("axis", "other")).strip() or "other",
                "points": r.get("points", 0),
            }
        )
    return out


def _infer_polarity(text: str, points: Any) -> str:
    t = str(text or "").lower()
    if isinstance(points, (int, float)) and points < 0:
        return "negative_constraint"
    neg_markers = ("避免", "禁止", "不得", "不应", "不能", "禁忌")
    if any(k in t for k in neg_markers):
        return "negative_constraint"
    return "positive_requirement"


def _fallback_atomic_points(rubrics: List[Dict]) -> List[Dict]:
    out = []
    for i, r in enumerate(rubrics, 1):
        out.append(
            {
                "text": r["criterion"],
                "source_criterion_index": i,
                "axis": r.get("axis", "other"),
                "polarity": _infer_polarity(r["criterion"], r.get("points")),
            }
        )
    return out


def _atomize_once(client: Any, conv_id: int, rubrics: List[Dict]) -> Tuple[List[Dict], str]:
    rubric_block = "\n".join(
        [
            f"{i}. [axis={r.get('axis','other')}, points={r.get('points')}] {r['criterion']}"
            for i, r in enumerate(rubrics, 1)
        ]
    )
    prompt = f"""
你是医学评估标准原子化助手。请将下列 rubric 条目分解为可二值判定的 Gold Atomic Points。

原子化要求：
1) 不可再分：每个点只包含一个可检查动作/事实/约束。
2) 自包含：不依赖代词或上下文歧义，单独存在即可判断。
3) 二值可验证：可直接判断“包含/不包含”。
4) 条件逻辑必须保留为整体（如“若X则Y”不可拆开）。
5) 禁忌/禁止/负向约束必须独立保留。
6) gold_atomic_points[].text 必须使用简体中文；可保留必要英文缩写（如 DCIS）。

输入 Rubrics（index={conv_id}）：
{rubric_block}

只输出 JSON：
{{
  "gold_atomic_points": [
    {{
      "text": "...",
      "source_criterion_index": 1,
      "axis": "accuracy|completeness|context_awareness|communication_quality|instruction_following|other",
      "polarity": "positive_requirement|negative_constraint"
    }}
  ]
}}
""".strip()

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
                return parsed.get("gold_atomic_points", []), raw
            last_error = "invalid_json_output"
        except Exception as e:
            last_error = _extract_error_message(e)
            print(f"[Atomize Retry {attempt + 1}] idx={conv_id} error: {last_error}")
            if _is_quota_error_text(last_error):
                raise RuntimeError(last_error) from e
            time.sleep(RETRY_SLEEP + attempt)
    raise RuntimeError(last_error or "atomize_failed")


def _sanitize_points(raw_points: List[Dict], rubrics: List[Dict]) -> List[Dict]:
    out = []
    seen = set()
    n = max(1, len(rubrics))
    for p in raw_points:
        if not isinstance(p, dict):
            continue
        src = p.get("source_criterion_index", 1)
        try:
            src = int(src)
        except Exception:
            src = 1
        if src < 1 or src > n:
            src = 1
        source_criterion = rubrics[src - 1].get("criterion", "")
        text = str(p.get("text", "")).strip()
        if not text:
            continue
        if not _has_cjk(text) and NON_CJK_STRATEGY == "source_fallback" and _has_cjk(source_criterion):
            text = str(source_criterion).strip()
        axis = str(p.get("axis", "")).strip() or rubrics[src - 1].get("axis", "other")
        polarity = str(p.get("polarity", "")).strip()
        if polarity not in ("positive_requirement", "negative_constraint"):
            polarity = _infer_polarity(text, rubrics[src - 1].get("points"))
        norm_key = re.sub(r"\s+", " ", text.lower())
        if norm_key in seen:
            continue
        seen.add(norm_key)
        out.append(
            {
                "text": text,
                "source_criterion_index": src,
                "axis": axis,
                "polarity": polarity,
            }
        )
    return out


def _rebuild_output_from_cache(rows: List[Dict]) -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for row in rows:
            try:
                idx = int(row.get("index"))
            except Exception:
                continue
            if not (START_INDEX <= idx <= END_INDEX):
                continue
            problem = str(row.get("problem", "")).strip()
            rubrics = _extract_rubrics(row.get("rubrics", []))
            if not rubrics:
                continue
            cache_file = CACHE_DIR / f"{idx}.json"
            if not cache_file.exists():
                continue
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                points = cached.get("gold_atomic_points", [])
            except Exception:
                points = []
            if not points:
                continue
            normalized_points = _sanitize_points(points, rubrics)
            if normalized_points and normalized_points != points:
                try:
                    cached["gold_atomic_points"] = normalized_points
                    cached["gold_atomic_count"] = len(normalized_points)
                    cache_file.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8")
                    points = normalized_points
                except Exception:
                    points = normalized_points

            enriched = []
            for i, p in enumerate(points, 1):
                src = p.get("source_criterion_index", 1)
                try:
                    src = int(src)
                except Exception:
                    src = 1
                if src < 1 or src > len(rubrics):
                    src = 1
                enriched.append(
                    {
                        "ga_id": f"{idx}_{i}",
                        "text": str(p.get("text", "")).strip(),
                        "axis": str(p.get("axis", "other")),
                        "polarity": str(p.get("polarity", "positive_requirement")),
                        "source_criterion_index": src,
                        "source_criterion": rubrics[src - 1]["criterion"] if rubrics else "",
                    }
                )

            rec = {
                "index": idx,
                "problem": problem,
                "source_rubric_count": len(rubrics),
                "gold_atomic_count": len(enriched),
                "gold_atomic_points": enriched,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    return written


def main():
    if not INPUT_PATH.exists():
        print(f"❌ Input not found: {INPUT_PATH}")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_jsonl(INPUT_PATH)
    if not rows:
        print(f"❌ No records loaded from {INPUT_PATH}")
        return

    client: Optional[Any] = None
    if API_KEY and OpenAI is not None:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60.0)
    elif API_KEY and OpenAI is None:
        print("⚠️ openai package not installed: will rebuild from cache only, and normalize non-Chinese atomic text using source criterion fallback.")
    else:
        print("⚠️ Missing API key: will rebuild from cache only, and normalize non-Chinese atomic text using source criterion fallback.")
    processed = 0
    cache_hit = 0
    fallback_used = 0
    skipped_no_api = 0

    for row in rows:
        try:
            idx = int(row.get("index"))
        except Exception:
            continue
        if not (START_INDEX <= idx <= END_INDEX):
            continue
        rubrics = _extract_rubrics(row.get("rubrics", []))
        if not rubrics:
            continue

        cache_file = CACHE_DIR / f"{idx}.json"
        if cache_file.exists() and cache_file.stat().st_size > 0:
            cache_hit += 1
            continue
        if client is None:
            skipped_no_api += 1
            continue

        print(f"🔬 Atomizing index={idx} ...", end="\r")
        raw_points = []
        raw_output = ""
        error = ""
        try:
            raw_points, raw_output = _atomize_once(client, idx, rubrics)
        except Exception as e:
            error = _extract_error_message(e)
            if _is_quota_error_text(error):
                print(f"\n🛑 Quota/limit reached at index={idx}: {error}")
                break
            raw_points = _fallback_atomic_points(rubrics)
            fallback_used += 1

        points = _sanitize_points(raw_points, rubrics)
        if not points:
            points = _fallback_atomic_points(rubrics)
            fallback_used += 1

        cache_obj = {
            "index": idx,
            "problem": str(row.get("problem", "")).strip(),
            "source_rubric_count": len(rubrics),
            "gold_atomic_count": len(points),
            "gold_atomic_points": points,
            "raw_output": raw_output,
            "error": error or None,
            "model": ATOMIZER_MODEL,
            "provider": API_PROVIDER,
        }
        cache_file.write_text(json.dumps(cache_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        processed += 1

    written = _rebuild_output_from_cache(rows)
    print("\n✅ Gold atomic points rebuilt from cache")
    print(f"   - input: {INPUT_PATH}")
    print(f"   - output: {OUTPUT_PATH} (rows={written})")
    print(f"   - cache: {CACHE_DIR}")
    print(f"   - new_processed: {processed}, cache_hit: {cache_hit}, fallback_used: {fallback_used}, skipped_no_api: {skipped_no_api}")


if __name__ == "__main__":
    main()

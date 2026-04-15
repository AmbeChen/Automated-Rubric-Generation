import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


START_INDEX = int(os.getenv("START_INDEX", 0))
END_INDEX = int(os.getenv("END_INDEX", 10**9))
MAX_RETRY = int(os.getenv("PERTURB_MAX_RETRY", 4))
RETRY_SLEEP = float(os.getenv("PERTURB_RETRY_SLEEP", 1.5))
TEMPERATURE = float(os.getenv("PERTURB_TEMPERATURE", 0.0))

BASE_DIR = Path(__file__).resolve().parent
RUBRICS_PATH = Path(os.getenv("LLM_RUBRICS_PATH", str(BASE_DIR / "outputs" / "rubrics_generated_zh.jsonl")))
REF_RESP_PATH = Path(os.getenv("REF_RESP_PATH", str(BASE_DIR / "outputs" / "eval" / "reference_responses_zh.jsonl")))
OUT_PATH = Path(os.getenv("PERTURBED_OUT_PATH", str(BASE_DIR / "outputs" / "eval" / "perturbed_candidates_zh.jsonl")))
FAIL_PATH = Path(os.getenv("PERTURBED_FAIL_PATH", str(BASE_DIR / "outputs" / "eval" / "perturbed_failures_zh.jsonl")))

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

MODEL = os.getenv("PERTURB_MODEL", os.getenv("JUDGE_MODEL", "gpt-4.1-mini"))

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

PROMPT = """
你是医学文本编辑器。请在“参考答案”基础上生成一个“最小扰动版本”。

目标：
1) 只引入一个细微但医学上有意义的错误（能违反目标要点）。
2) 除这一个错误外，其余内容尽量保持不变。

硬性要求：
- 只改一个连续片段。
- 不要改写整段风格。
- 不要增加额外解释文本。
- 只输出 JSON。

输入：
[目标要点]
{target}

[问题]
{question}

[参考答案]
{reference}

输出 JSON：
{{
  "edit_span_before": "...",
  "edit_span_after": "...",
  "perturbed_response": "...",
  "error_type": "numeric_threshold|modality_flip|exception_scope|fact_swap|other"
}}
""".strip()


def _is_quota_error_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    s = text.lower()
    return any(k in s for k in QUOTA_ERROR_KEYWORDS)


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
    if not isinstance(text, str):
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t.replace("json", "", 1).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", t)
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
        try:
            pts = float(r.get("points", 0))
        except Exception:
            pts = 0.0
        out.append(
            {
                "criterion": c,
                "axis": str(r.get("axis", "completeness")).strip() or "completeness",
                "points": pts,
            }
        )
    return out


def _pick_target(rubrics: List[Dict]) -> str:
    if not rubrics:
        return ""
    def score(x):
        bonus = 3.0 if x.get("axis") == "accuracy" else 0.0
        return bonus + float(x.get("points", 0))
    s = sorted(rubrics, key=score, reverse=True)
    return str(s[0].get("criterion", "")).strip()


def _minimal_edit_check(ref: str, pert: str, before: str, after: str) -> bool:
    if not before or before not in ref:
        return False
    if not after or not pert:
        return False
    cand = ref.replace(before, after, 1)
    norm = lambda s: re.sub(r"\s+", " ", str(s).strip())
    return norm(cand) == norm(pert)


def _rule_fallback(ref: str):
    m = re.search(r"\b\d+(?:\.\d+)?\b", ref)
    if m:
        b = m.group(0)
        if "." in b:
            try:
                a = str(round(float(b) + 0.5, 2)).rstrip("0").rstrip(".")
            except Exception:
                a = b + "1"
        else:
            try:
                a = str(int(b) + 1)
            except Exception:
                a = b + "1"
        return b, a, ref.replace(b, a, 1), "numeric_threshold"

    pairs = [
        ("应该", "不应该"),
        ("可", "不可"),
        ("可以", "不可以"),
        ("建议", "不建议"),
    ]
    for b, a in pairs:
        if b in ref:
            return b, a, ref.replace(b, a, 1), "modality_flip"
    return "", "", ref, "other"


def main():
    if not API_KEY:
        print("❌ Missing API key.")
        return
    if not RUBRICS_PATH.exists() or not REF_RESP_PATH.exists():
        print(f"❌ Missing input file: rubrics={RUBRICS_PATH.exists()} refs={REF_RESP_PATH.exists()}")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FAIL_PATH.parent.mkdir(parents=True, exist_ok=True)

    rubrics_rows = _load_jsonl(RUBRICS_PATH)
    ref_rows = _load_jsonl(REF_RESP_PATH)
    ref_map = {}
    for r in ref_rows:
        try:
            idx = int(r.get("index"))
        except Exception:
            continue
        ref_map[idx] = r

    done = set()
    if OUT_PATH.exists():
        for r in _load_jsonl(OUT_PATH):
            try:
                done.add(int(r.get("index")))
            except Exception:
                pass
        print(f"[Resume] already done: {len(done)}")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60.0)
    written = 0
    failed = 0

    with OUT_PATH.open("a", encoding="utf-8") as fout, FAIL_PATH.open("a", encoding="utf-8") as ffail:
        for row in rubrics_rows:
            try:
                idx = int(row.get("index"))
            except Exception:
                continue
            if idx in done:
                continue
            if not (START_INDEX <= idx <= END_INDEX):
                continue

            problem = str(row.get("problem", "")).strip()
            rubrics = _extract_rubrics(row.get("generated_rubrics", []))
            target = _pick_target(rubrics)
            ref = str((ref_map.get(idx) or {}).get("reference_response", "")).strip()
            if not problem or not target or not ref:
                failed += 1
                ffail.write(
                    json.dumps(
                        {"index": idx, "problem": problem, "reason": "missing_problem_or_target_or_ref"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            success = False
            for attempt in range(1, MAX_RETRY + 1):
                prompt = PROMPT.format(target=target, question=problem, reference=ref)
                try:
                    resp = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=TEMPERATURE,
                    )
                    raw = (resp.choices[0].message.content or "").strip()
                except Exception as e:
                    msg = _extract_error_message(e)
                    if _is_quota_error_text(msg):
                        print(f"\n🛑 Quota exhausted at index={idx}: {msg}")
                        return
                    if attempt < MAX_RETRY:
                        time.sleep(RETRY_SLEEP * attempt)
                    continue

                parsed = _safe_parse_json(raw)
                if not isinstance(parsed, dict):
                    if attempt < MAX_RETRY:
                        time.sleep(RETRY_SLEEP * attempt)
                    continue

                before = str(parsed.get("edit_span_before", "")).strip()
                after = str(parsed.get("edit_span_after", "")).strip()
                pert = str(parsed.get("perturbed_response", "")).strip()
                err_type = str(parsed.get("error_type", "other")).strip() or "other"
                strict_ok = _minimal_edit_check(ref, pert, before, after)

                if not pert:
                    if attempt < MAX_RETRY:
                        time.sleep(RETRY_SLEEP * attempt)
                    continue

                out = {
                    "index": idx,
                    "problem": problem,
                    "target_keypoint": target,
                    "edit_span_before": before,
                    "edit_span_after": after,
                    "perturbed_response": pert,
                    "error_type": err_type,
                    "is_strict_minimal_edit": strict_ok,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                written += 1
                success = True
                print(f"[OK] idx={idx} strict_edit={strict_ok}")
                break

            if not success:
                b, a, p, et = _rule_fallback(ref)
                out = {
                    "index": idx,
                    "problem": problem,
                    "target_keypoint": target,
                    "edit_span_before": b,
                    "edit_span_after": a,
                    "perturbed_response": p,
                    "error_type": et,
                    "is_strict_minimal_edit": _minimal_edit_check(ref, p, b, a) if b else False,
                    "fallback": True,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                written += 1
                print(f"[Fallback] idx={idx}")

    print("\n✅ Perturbed generation done")
    print(f"   - output: {OUT_PATH}")
    print(f"   - failures: {FAIL_PATH}")
    print(f"   - written={written}, failed={failed}")


if __name__ == "__main__":
    main()

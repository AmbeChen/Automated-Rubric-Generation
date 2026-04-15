import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List


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
                "axis": str(r.get("axis", "completeness")).strip() or "completeness",
                "points": r.get("points", 6),
            }
        )
    return out


def _norm_text(s: str) -> str:
    t = str(s or "").strip().lower()
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"[，。！？；：、“”‘’（）()【】\\[\\],.!?:;\"'`~·-]", "", t)
    return t


def main():
    base = Path(__file__).resolve().parent
    dataset_path = base / "dataset_LLMEval_Med.jsonl"
    generated_path = base / "outputs" / "rubrics_generated_zh.jsonl"
    reference_path = base / "outputs" / "reference_rubrics_from_checklist_zh.jsonl"
    out_dir = base / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_ref_resp = out_dir / "reference_responses_zh.jsonl"
    out_gold_rubrics = out_dir / "gold_reference_rubrics_zh.jsonl"
    out_index_map = out_dir / "index_problem_answer_map_zh.jsonl"

    ds_rows = _load_jsonl(dataset_path)
    gen_rows = _load_jsonl(generated_path)
    ref_rows = _load_jsonl(reference_path)

    problem_to_answer: Dict[str, str] = {}
    dataset_problem_list: List[str] = []
    for r in ds_rows:
        p = str(r.get("problem", "")).strip()
        a = str(r.get("sanswer", "")).strip()
        if p and a and p not in problem_to_answer:
            problem_to_answer[p] = a
            dataset_problem_list.append(p)

    norm_problem_to_answer: Dict[str, str] = {}
    norm_to_raw: Dict[str, str] = {}
    for p, a in problem_to_answer.items():
        n = _norm_text(p)
        if n and n not in norm_problem_to_answer:
            norm_problem_to_answer[n] = a
            norm_to_raw[n] = p

    ref_map: Dict[int, Dict] = {}
    for r in ref_rows:
        try:
            idx = int(r.get("index"))
        except Exception:
            continue
        ref_map[idx] = r

    written_ref = 0
    written_gold = 0
    missing_answer = 0
    match_stats = {"exact": 0, "normalized": 0, "fuzzy": 0, "missing": 0}

    with out_ref_resp.open("w", encoding="utf-8") as f_ref, out_gold_rubrics.open(
        "w", encoding="utf-8"
    ) as f_gold, out_index_map.open("w", encoding="utf-8") as f_map:
        for row in gen_rows:
            try:
                idx = int(row.get("index"))
            except Exception:
                continue
            problem = str(row.get("problem", "")).strip()
            if not problem:
                continue

            ans = ""
            match_type = "missing"
            matched_problem = ""
            if problem in problem_to_answer:
                ans = problem_to_answer[problem]
                match_type = "exact"
                matched_problem = problem
            else:
                n = _norm_text(problem)
                if n in norm_problem_to_answer:
                    ans = norm_problem_to_answer[n]
                    match_type = "normalized"
                    matched_problem = norm_to_raw.get(n, "")
                else:
                    # fuzzy fallback for "该药/该手术/该疾病" 这类改写
                    best_ratio = 0.0
                    best_problem = ""
                    for dp in dataset_problem_list:
                        ratio = SequenceMatcher(None, problem, dp).ratio()
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_problem = dp
                    if best_problem and best_ratio >= 0.70:
                        ans = problem_to_answer.get(best_problem, "")
                        if ans:
                            match_type = "fuzzy"
                            matched_problem = best_problem
            if not ans:
                missing_answer += 1
                match_stats["missing"] += 1
            else:
                match_stats[match_type] += 1
                f_ref.write(
                    json.dumps(
                        {
                            "index": idx,
                            "problem": problem,
                            "reference_response": ans,
                            "matched_problem": matched_problem or problem,
                            "match_type": match_type,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written_ref += 1

            ref_obj = ref_map.get(idx, {})
            gold_rubrics = _extract_rubrics(ref_obj.get("reference_rubrics", []))
            if gold_rubrics:
                f_gold.write(
                    json.dumps(
                        {
                            "index": idx,
                            "problem": problem,
                            "rubrics": gold_rubrics,
                            "source_rubric_count": len(gold_rubrics),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written_gold += 1

            f_map.write(
                json.dumps(
                    {
                        "index": idx,
                        "problem": problem,
                        "has_reference_response": bool(ans),
                        "has_reference_rubrics": bool(gold_rubrics),
                        "match_type": match_type,
                        "matched_problem": matched_problem or None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("✅ Prepared LLMEval_Med evaluation assets")
    print(f"   - reference responses: {out_ref_resp} (rows={written_ref})")
    print(f"   - gold reference rubrics: {out_gold_rubrics} (rows={written_gold})")
    print(f"   - index map: {out_index_map}")
    print(f"   - missing answers by problem match: {missing_answer}")
    print(f"   - match stats: {match_stats}")


if __name__ == "__main__":
    main()

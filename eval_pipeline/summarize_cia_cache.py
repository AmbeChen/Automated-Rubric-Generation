import json
import math
import os
from argparse import ArgumentParser
from datetime import datetime, timezone
from typing import Dict, List


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def wilson_ci_95(yes: int, total: int):
    if total <= 0:
        return {"low": None, "high": None}
    z = 1.959963984540054
    p = yes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total) / denom
    low = max(0.0, center - half) * 100
    high = min(1.0, center + half) * 100
    return {"low": low, "high": high}


def two_proportion_pvalue(x1: int, n1: int, x2: int, n2: int):
    if n1 <= 0 or n2 <= 0:
        return None
    p_pool = (x1 + x2) / (n1 + n2)
    var = p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)
    if var <= 0:
        p1 = x1 / n1
        p2 = x2 / n2
        return 1.0 if abs(p1 - p2) < 1e-12 else 0.0
    z = (x1 / n1 - x2 / n2) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2.0))


def parse_modes(cache_root: str, raw_modes: str) -> List[str]:
    if raw_modes.strip():
        return [x.strip() for x in raw_modes.split(",") if x.strip()]
    if not os.path.isdir(cache_root):
        return []
    modes = []
    for name in sorted(os.listdir(cache_root)):
        p = os.path.join(cache_root, name)
        if os.path.isdir(p):
            modes.append(name)
    return modes


def summarize_mode(cache_dir: str) -> Dict:
    total = 0
    yes = 0
    files = 0
    invalid_files = 0

    if not os.path.isdir(cache_dir):
        return {
            "cached_files": 0,
            "invalid_files": 0,
            "total_gold_atomic_points": 0,
            "count_yes": 0,
            "cia_score_percent": 0.0,
            "ci_95_percent": {"low": None, "high": None},
        }

    for name in sorted(os.listdir(cache_dir)):
        if not name.endswith(".json"):
            continue
        p = os.path.join(cache_dir, name)
        files += 1
        try:
            obj = json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            invalid_files += 1
            continue

        records = obj.get("atomic_results", [])
        if not isinstance(records, list):
            invalid_files += 1
            continue

        for r in records:
            total += 1
            if str((r or {}).get("decision", "")).upper() == "YES":
                yes += 1

    score = (yes / total * 100) if total else 0.0
    return {
        "cached_files": files,
        "invalid_files": invalid_files,
        "total_gold_atomic_points": total,
        "count_yes": yes,
        "cia_score_percent": score,
        "ci_95_percent": wilson_ci_95(yes, total),
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--modes", default="")
    parser.add_argument("--pvalue-baseline", default="gpt4o")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--api-provider", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cache_root = args.cache_root
    modes = parse_modes(cache_root, args.modes)
    if not modes:
        print("❌ No modes found to summarize.")
        return

    if args.output.strip():
        output_path = args.output.strip()
    else:
        output_path = os.path.join(os.path.dirname(cache_root), "cia_cache_summary_compact.json")

    summary = {
        "script": "eval_pipeline/summarize_cia_cache.py",
        "generated_at": utc_now(),
        "cache_root": cache_root,
        "judge_model": args.judge_model or None,
        "api_provider": args.api_provider or None,
        "modes_requested": modes,
        "p_value_baseline_mode": args.pvalue_baseline,
        "modes": [],
    }

    for mode in modes:
        mode_dir = os.path.join(cache_root, mode)
        stats = summarize_mode(mode_dir)
        stats["mode"] = mode
        summary["modes"].append(stats)

    baseline = None
    for m in summary["modes"]:
        if m.get("mode") == args.pvalue_baseline:
            baseline = m
            break

    p_key = f"p_value_vs_{args.pvalue_baseline}"
    for m in summary["modes"]:
        if baseline is None:
            m[p_key] = None
        elif m["mode"] == args.pvalue_baseline:
            m[p_key] = 1.0
        else:
            m[p_key] = two_proportion_pvalue(
                int(m.get("count_yes", 0)),
                int(m.get("total_gold_atomic_points", 0)),
                int(baseline.get("count_yes", 0)),
                int(baseline.get("total_gold_atomic_points", 0)),
            )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ CIA cache compact summary saved to: {output_path}")


if __name__ == "__main__":
    main()

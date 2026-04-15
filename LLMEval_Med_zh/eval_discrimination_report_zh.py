import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score


def _normalize_one(parsed: Dict, a_is_reference: bool):
    if not isinstance(parsed, dict):
        return None
    dec = parsed.get("decision")
    total = parsed.get("total", {})
    if not isinstance(total, dict):
        total = {}

    if a_is_reference:
        score_ref = total.get("A", 0)
        score_cand = total.get("B", 0)
        if dec == "A":
            decision = "REF"
        elif dec == "B":
            decision = "CAND"
        else:
            decision = "SAME"
    else:
        score_ref = total.get("B", 0)
        score_cand = total.get("A", 0)
        if dec == "B":
            decision = "REF"
        elif dec == "A":
            decision = "CAND"
        else:
            decision = "SAME"

    try:
        delta = float(score_ref) - float(score_cand)
    except Exception:
        delta = 0.0
    return {"decision": decision, "delta": delta}


def _majority_vote(decisions: List[str]) -> str:
    c_ref = sum(1 for d in decisions if d == "REF")
    c_cand = sum(1 for d in decisions if d == "CAND")
    c_same = sum(1 for d in decisions if d == "SAME")
    if c_ref > max(c_cand, c_same):
        return "REF"
    if c_cand > max(c_ref, c_same):
        return "CAND"
    return "SAME"


def _bootstrap_ci(values: List[float], n_boot=1000, alpha=0.05):
    if not values:
        return None, None
    arr = list(values)
    n = len(arr)
    stats = []
    for _ in range(n_boot):
        sample = [arr[random.randrange(n)] for _ in range(n)]
        stats.append(float(np.mean(sample)))
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lo, hi


def evaluate_file(path: Path) -> Dict:
    if not path.exists():
        return {"status": "missing", "path": str(path)}

    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            results = obj.get("results", [])
            per_call = []
            for r in results:
                if not isinstance(r, dict):
                    continue
                norm = _normalize_one(r.get("parsed"), bool(r.get("A_text")))
                if norm:
                    per_call.append(norm)
            if not per_call:
                continue
            final_dec = _majority_vote([x["decision"] for x in per_call])
            mean_delta = float(np.mean([x["delta"] for x in per_call]))
            pairs.append({"pair_id": obj.get("pair_id"), "decision": final_dec, "delta": mean_delta})

    n = len(pairs)
    if n == 0:
        return {"status": "empty", "path": str(path), "n_pairs": 0}

    wins = sum(1 for p in pairs if p["decision"] == "REF")
    losses = sum(1 for p in pairs if p["decision"] == "CAND")
    ties = sum(1 for p in pairs if p["decision"] == "SAME")

    win_rate = wins / n
    loss_rate = losses / n
    tie_rate = ties / n
    deltas = [float(p["delta"]) for p in pairs]
    mean_delta = float(np.mean(deltas))
    median_delta = float(np.median(deltas))
    win_ci = _bootstrap_ci([1.0 if p["decision"] == "REF" else 0.0 for p in pairs])
    delta_ci = _bootstrap_ci(deltas)

    y_true = np.array([1 if p["decision"] == "REF" else 0 for p in pairs])
    y_score = np.array(deltas, dtype=float)
    auroc = None
    if len(set(y_true.tolist())) > 1:
        auroc = float(roc_auc_score(y_true, y_score))

    return {
        "status": "ok",
        "path": str(path),
        "n_pairs": n,
        "win_rate_ref": win_rate,
        "loss_rate_ref": loss_rate,
        "tie_rate": tie_rate,
        "win_rate_ci_95": {"low": win_ci[0], "high": win_ci[1]},
        "mean_delta_ref_minus_cand": mean_delta,
        "median_delta_ref_minus_cand": median_delta,
        "mean_delta_ci_95": {"low": delta_ci[0], "high": delta_ci[1]},
        "auroc": auroc,
    }


def main():
    ap = argparse.ArgumentParser()
    default_dir = Path(__file__).resolve().parent / "outputs" / "eval" / "discrimination"
    ap.add_argument("--input-dir", default=str(default_dir))
    ap.add_argument("--modes", default="ours,none")
    ap.add_argument(
        "--output",
        default=str(default_dir / "final_report_summary_zh.json"),
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    summary = {
        "script": "LLMEval_Med/eval_discrimination_report_zh.py",
        "input_dir": str(input_dir),
        "modes_requested": modes,
        "modes": [],
    }

    for mode in modes:
        result_file = input_dir / f"results_{mode}.jsonl"
        stats = evaluate_file(result_file)
        stats["mode"] = mode
        summary["modes"].append(stats)
        if stats.get("status") == "ok":
            print(
                f"[{mode}] N={stats['n_pairs']} "
                f"win={stats['win_rate_ref']:.3f} "
                f"delta={stats['mean_delta_ref_minus_cand']:.3f} "
                f"auroc={stats['auroc'] if stats['auroc'] is not None else 'NA'}"
            )
        else:
            print(f"[{mode}] status={stats.get('status')}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Saved: {out}")


if __name__ == "__main__":
    main()

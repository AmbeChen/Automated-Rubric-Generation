import json
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import argparse
import random

# ======================
# Utils
# ======================

def normalize_one(parsed, A_is_reference):
    """
    统一成 reference vs candidate 视角
    返回:
      decision: REF / CAND / SAME
      delta: score_ref - score_cand
    """
    if parsed is None:
        return None

    dec = parsed.get("decision")
    total = parsed.get("total", {})

    if A_is_reference:
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

    return {
        "decision": decision,
        "delta": score_ref - score_cand
    }


def majority_vote(decisions):
    cnt = defaultdict(int)
    for d in decisions:
        cnt[d] += 1
    if cnt["REF"] > max(cnt["CAND"], cnt["SAME"]):
        return "REF"
    if cnt["CAND"] > max(cnt["REF"], cnt["SAME"]):
        return "CAND"
    return "SAME"


def bootstrap_ci(values, fn=np.mean, n_boot=1000, alpha=0.05):
    stats = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[random.randrange(n)] for _ in range(n)]
        stats.append(fn(sample))
    lo = np.percentile(stats, 100 * alpha / 2)
    hi = np.percentile(stats, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


# ======================
# Main
# ======================

def main(input_file, assume_ref_better=True):
    """
    assume_ref_better:
      True  -> reference 应该优于 candidate（reference vs perturbed）
      False -> 不设 ground truth，只做描述统计
    """

    all_pairs = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            per_call = []
            for r in obj["results"]:
                norm = normalize_one(r["parsed"], r["A_text"])
                if norm:
                    per_call.append(norm)

            if not per_call:
                continue

            final_decision = majority_vote([x["decision"] for x in per_call])
            mean_delta = np.mean([x["delta"] for x in per_call])

            all_pairs.append({
                "pair_id": obj["pair_id"],
                "decision": final_decision,
                "delta": mean_delta
            })

    # ======================
    # Pairwise metrics
    # ======================

    n = len(all_pairs)
    wins = sum(1 for x in all_pairs if x["decision"] == "REF")
    losses = sum(1 for x in all_pairs if x["decision"] == "CAND")
    ties = sum(1 for x in all_pairs if x["decision"] == "SAME")

    win_rate = wins / n
    loss_rate = losses / n
    tie_rate = ties / n

    print("===== PAIRWISE =====")
    print(f"N = {n}")
    print(f"Win (REF):  {win_rate:.3f}")
    print(f"Loss:       {loss_rate:.3f}")
    print(f"Tie:        {tie_rate:.3f}")

    # Bootstrap CI
    win_flags = [1 if x["decision"] == "REF" else 0 for x in all_pairs]
    win_ci = bootstrap_ci(win_flags, np.mean)

    print(f"Win-rate 95% CI: [{win_ci[0]:.3f}, {win_ci[1]:.3f}]")

    # ======================
    # Score delta stats
    # ======================

    deltas = np.array([x["delta"] for x in all_pairs])

    print("\n===== SCORE DELTA =====")
    print(f"Mean Δ:   {deltas.mean():.3f}")
    print(f"Median Δ:{np.median(deltas):.3f}")

    delta_ci = bootstrap_ci(deltas, np.mean)
    print(f"Mean Δ 95% CI: [{delta_ci[0]:.3f}, {delta_ci[1]:.3f}]")

    # ======================
    # AUROC (if GT exists)
    # ======================

    if assume_ref_better:
        # 正类 = REF 胜
        y_true = np.array([1 if x["decision"] == "REF" else 0 for x in all_pairs])
        y_score = deltas

        if len(set(y_true)) > 1:
            auc = roc_auc_score(y_true, y_score)
            print("\n===== AUROC =====")
            print(f"AUROC: {auc:.3f}")
        else:
            print("\n[Warning] AUROC undefined (only one class).")

    print("\n===== DONE =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--assume_ref_better", action="store_true")
    args = parser.parse_args()

    main(args.input, assume_ref_better=args.assume_ref_better)

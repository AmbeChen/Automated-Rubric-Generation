import argparse
import json
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


def _build_map(rows: List[Dict], idx_key="index") -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    for r in rows:
        try:
            idx = int(r.get(idx_key))
        except Exception:
            continue
        out[idx] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent / "outputs" / "eval"
    ap.add_argument("--rubrics", default=str(Path(__file__).resolve().parent / "outputs" / "rubrics_generated_zh.jsonl"))
    ap.add_argument("--reference", default=str(base / "reference_responses_zh.jsonl"))
    ap.add_argument("--perturbed", default=str(base / "perturbed_candidates_zh.jsonl"))
    ap.add_argument("--output", default=str(base / "pairwise_dataset_perturbed_zh.jsonl"))
    ap.add_argument("--missing-log", default=str(base / "pairwise_dataset_perturbed_zh_missing.jsonl"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10**9)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    rubrics_rows = _load_jsonl(Path(args.rubrics))
    ref_map = _build_map(_load_jsonl(Path(args.reference)))
    per_map = _build_map(_load_jsonl(Path(args.perturbed)))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    miss_path = Path(args.missing_log)
    miss_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and out_path.exists():
        for r in _load_jsonl(out_path):
            pid = str(r.get("pair_id", "")).strip()
            if pid:
                done.add(pid)
        print(f"[Resume] existing pairs: {len(done)}")

    written = 0
    skipped = 0
    missing_rows = []

    with out_path.open("a" if args.resume else "w", encoding="utf-8") as fout:
        for r in rubrics_rows:
            try:
                idx = int(r.get("index"))
            except Exception:
                continue
            if not (args.start <= idx <= args.end):
                continue
            pair_id = f"{idx:06d}"
            if pair_id in done:
                continue

            question = str(r.get("problem", "")).strip()
            ref = str((ref_map.get(idx) or {}).get("reference_response", "")).strip()
            cand = str((per_map.get(idx) or {}).get("perturbed_response", "")).strip()

            if not question or not ref or not cand:
                skipped += 1
                missing_rows.append(
                    {
                        "index": idx,
                        "pair_id": pair_id,
                        "has_question": bool(question),
                        "has_ref": bool(ref),
                        "has_candidate": bool(cand),
                    }
                )
                continue

            obj = {
                "pair_id": pair_id,
                "conv_id": idx,
                "question": question,
                "A": {"text": ref, "system_id": "reference"},
                "B": {"text": cand, "system_id": "candidate"},
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
            if written % 50 == 0:
                print(f"[OK] written={written}")

    with miss_path.open("w", encoding="utf-8") as mf:
        for m in missing_rows:
            mf.write(json.dumps(m, ensure_ascii=False) + "\n")

    print("✅ Pairwise dataset built")
    print(f"   - output: {out_path}")
    print(f"   - missing: {miss_path}")
    print(f"   - written={written}, skipped_missing={skipped}")


if __name__ == "__main__":
    main()

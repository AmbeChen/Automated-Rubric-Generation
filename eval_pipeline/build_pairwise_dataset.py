import os
import json
import argparse

def load_json_or_jsonl_to_map(path, key_field="index", value_field=None):
    """
    兼容：
    1) JSONL（每行一个 JSON）
    2) 整体 JSON list
    3) 整体 JSON dict
    返回 dict[key] = record or record[value_field]
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    data = []

    # ---------- 尝试整体 JSON ----------
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            data = obj
        elif isinstance(obj, dict):
            data = list(obj.values())
    except json.JSONDecodeError:
        pass

    # ---------- 如果整体 JSON 失败，按 JSONL ----------
    if not data:
        data = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print("[Warning] Skip unparsable line:", line[:80])

    # ---------- 构建 map ----------
    m = {}
    for obj in data:
        if not isinstance(obj, dict):
            continue
        idx = obj.get(key_field)
        if idx is None:
            continue
        if value_field is None:
            m[idx] = obj
        else:
            m[idx] = obj.get(value_field, "")
    return m


def load_conversations_txt(path):
    """
    conversations_all.txt 每行形如: 0\tUser: ....
    返回 dict[index] = question
    """
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            # 兼容: "0\tUser: xxx"
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            try:
                idx = int(parts[0].strip())
            except:
                continue
            text = parts[1].strip()
            # 去掉 "User:" 前缀（如果有）
            if text.lower().startswith("user:"):
                text = text[5:].strip()
            m[idx] = text
    return m

def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    rubrics_map = load_json_or_jsonl_to_map(
        args.rubrics_file, key_field="index"
    )

    ref_map = load_json_or_jsonl_to_map(
        args.reference_file,
        key_field="index",
        value_field="reference_response"
    )

    cand_map = load_json_or_jsonl_to_map(
        args.candidate_file,
        key_field="index",
        value_field="perturbed_response"
    )

    # 4) optional conversations txt (fallback question)
    conv_map = load_conversations_txt(args.conversations_file) if args.conversations_file else {}

    # resume: 已经写过的 pair_id 不再写
    done = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    done.add(obj.get("pair_id"))
                except:
                    continue
        print(f"[Resume] found {len(done)} existing pairs in {args.output}")

    written = 0
    skipped = 0

    mode = args.mode.lower()
    assert mode in ["ref_vs_perturbed", "ref_vs_model"], "mode must be ref_vs_perturbed or ref_vs_model"

    # 以 rubrics 为主索引集合（最稳：rubric 缺了就没法评）
    indices = sorted(rubrics_map.keys())

    with open(args.output, "a" if args.resume else "w", encoding="utf-8") as fout:
        for idx in indices:
            if not (args.start <= idx < args.end):
                continue

            pair_id = f"{idx:06d}"
            if pair_id in done:
                continue

            r = rubrics_map.get(idx, {})
            question = (r.get("original_query") or "").strip()
            if not question:
                question = (conv_map.get(idx) or "").strip()

            ref = (ref_map.get(idx) or "").strip()
            cand = (cand_map.get(idx) or "").strip()

            if not question or not ref or not cand:
                skipped += 1
                # 只打印少量 warning，避免刷屏
                if skipped <= 20:
                    print(f"[Skip] idx={idx} missing field(s): "
                          f"question={'Y' if question else 'N'}, ref={'Y' if ref else 'N'}, cand={'Y' if cand else 'N'}")
                continue

            # A 永远是 reference；B 是 candidate（perturbed 或 model）
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

            if written % 100 == 0:
                print(f"[OK] written {written} pairs")

    print(f"Done. written={written}, skipped={skipped}, output={args.output}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubrics_file", required=True)
    ap.add_argument("--reference_file", required=True)
    ap.add_argument("--candidate_file", required=True)
    ap.add_argument("--conversations_file", default="")
    ap.add_argument("--output", required=True)

    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10**9)

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--mode", default="ref_vs_perturbed",
                    help="ref_vs_perturbed (default) or ref_vs_model (same schema, just naming)")

    args = ap.parse_args()
    main(args)

import json
import argparse
import time
from pathlib import Path

from utils.llm_client import get_client
from utils.robust_json import try_parse_json   # 就是你贴的那个

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def pick(d, keys, default=""):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=["groq", "cerebras"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=600)
    parser.add_argument("--retries", type=int, default=3)

    parser.add_argument("--base", default="data/responses_llama3_70b.jsonl")
    parser.add_argument("--refined", default="outputs/70b/refined_responses.jsonl")
    parser.add_argument("--rubrics", default="data/reference_rubrics_converted.jsonl")
    parser.add_argument("--out", default="outputs/70b/final_scores.jsonl")
    parser.add_argument("--prompt", default="prompts/healthbench_scoring_prompt.txt")
    args = parser.parse_args()

    client = get_client(args.provider)

    base_rows = load_jsonl(Path(args.base))
    refined_rows = load_jsonl(Path(args.refined))
    rubric_rows = load_jsonl(Path(args.rubrics))

    base_by_id = {int(r["id"]): r for r in base_rows}
    refined_by_id = {int(r["id"]): r for r in refined_rows}
    rubric_by_id = {}
    for r in rubric_rows:
        if "id" in r:
            key = int(r["id"])
        elif "index" in r:
            key = int(r["index"])
        else:
            raise KeyError("Rubric row missing both 'id' and 'index'")
        rubric_by_id[key] = r
    
    print(
    f"Loaded base={len(base_by_id)}, refined={len(refined_by_id)}, rubrics={len(rubric_by_id)}",
    flush=True
)



    ids = sorted(set(base_by_id) & set(refined_by_id) & set(rubric_by_id))
    if args.end is None:
        args.end = len(ids)
    ids = ids[args.start:args.end]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            obj = json.loads(line)
            done.add((obj["id"], obj["variant"]))

    system_prompt = Path(args.prompt).read_text(encoding="utf-8")

    def score_once(question, response, rubric):
        user_msg = f"""
Question:
{question}

Response:
{response}

Rubric:
{json.dumps(rubric, ensure_ascii=False)}
"""
        last_err = None
        for _ in range(args.retries):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    temperature=0,
                    max_tokens=args.max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                )
                raw = resp.choices[0].message.content
                obj, reason = try_parse_json(raw)
                if obj is not None and "overall_score" in obj:
                    return obj
                last_err = reason
            except Exception as e:
                last_err = str(e)
            time.sleep(1)
        raise RuntimeError(f"Judge failed: {last_err}")

    with open(out_path, "a", encoding="utf-8") as fout:
        for i, rid in enumerate(ids, 1):
            base = base_by_id[rid]
            refined = refined_by_id[rid]
            rubric = rubric_by_id[rid].get("rubric", rubric_by_id[rid])

            question = pick(base, ["query", "question", "prompt"])
            base_resp = pick(base, ["answer", "response"])
            refined_resp = refined["refined_response"]

            for variant, resp_text in [
                ("base", base_resp),
                ("refined", refined_resp),
            ]:
                if (rid, variant) in done:
                    continue

                print(f"[{i}/{len(ids)}] id={rid} {variant} scoring...", flush=True)

                score = score_once(question, resp_text, rubric)

                fout.write(json.dumps({
                    "id": rid,
                    "variant": variant,
                    "overall_score": int(score["overall_score"]),
                    "axis_scores": score.get("axis_scores", {}),
                    "judge_model": args.model
                }, ensure_ascii=False) + "\n")
                fout.flush()

    print("✔ HealthBench scoring finished.")

if __name__ == "__main__":
    main()

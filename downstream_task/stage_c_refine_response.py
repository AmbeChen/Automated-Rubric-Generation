import json
import argparse
import time
from pathlib import Path

from utils.llm_client import get_client

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
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--provider", type=str, default="groq")
    parser.add_argument("--model", type=str, default="llama-3.1-8b-instant")
    parser.add_argument("--max_tokens", type=int, default=900)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    client = get_client(args.provider)

    RESP_PATH = Path("data/responses_llama3_8b.jsonl")
    CRIT_PATH = Path("outputs/70b/critique.jsonl")
    PROMPT = Path("prompts/rewrite_prompt.txt").read_text()

    OUT_FILE = Path("outputs/70b/refined_responses.jsonl")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    responses = load_jsonl(RESP_PATH)
    critiques = load_jsonl(CRIT_PATH)

    # -------- id -> response 映射 --------
    resp_by_id = {int(r["id"]): r for r in responses if "id" in r}

    # -------- 断点恢复 --------
    done_ids = set()
    if OUT_FILE.exists():
        with open(OUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                done_ids.add(int(obj["id"]))

    end = args.end if args.end is not None else len(critiques)
    total = min(end, len(critiques)) - args.start
    if total <= 0:
        print("Nothing to do.")
        return

    print(f"▶ Refinement start: {total} items")

    with open(OUT_FILE, "a", encoding="utf-8") as fout:
        done_count = 0

        for i in range(args.start, min(end, len(critiques))):
            citem = critiques[i]
            rid = int(citem["id"])

            if rid in done_ids:
                continue

            resp_item = resp_by_id.get(rid)
            if resp_item is None:
                print(f"⚠️ Skip id={rid}: response not found")
                continue

            question = pick(resp_item, ["query", "question", "prompt", "conversation"])
            base_answer = pick(resp_item, ["answer", "response", "output"])

            critique = citem["critique"]
            edit_plan = critique.get("edit_plan", [])
            constraints = critique.get("constraints", [])

            rewrite_prompt = PROMPT.format(
                question=question,
                response=base_answer,
                edit_plan=json.dumps(edit_plan, ensure_ascii=False, indent=2),
                constraints=json.dumps(constraints, ensure_ascii=False, indent=2)
            )

            print(f"[{done_count+1}/{total}] id={rid} rewriting...", flush=True)

            last_err = None
            for attempt in range(1, args.retries + 1):
                try:
                    t0 = time.time()
                    llm_resp = client.chat.completions.create(
                        model=args.model,
                        temperature=0.2,
                        max_tokens=args.max_tokens,
                        messages=[{"role": "user", "content": rewrite_prompt}],
                    )
                    refined = (llm_resp.choices[0].message.content or "").strip()
                    if not refined:
                        raise ValueError("Empty model output")

                    fout.write(json.dumps({
                        "id": rid,
                        "base_response": base_answer,
                        "refined_response": refined,
                        "model": args.model
                    }, ensure_ascii=False) + "\n")
                    fout.flush()

                    dt = time.time() - t0
                    print(f"[{done_count+1}/{total}] id={rid} ✅ done in {dt:.2f}s", flush=True)

                    done_ids.add(rid)
                    done_count += 1
                    last_err = None
                    break

                except Exception as e:
                    last_err = e
                    print(f"  ⚠️ attempt {attempt}/{args.retries} failed: {e}", flush=True)
                    time.sleep(1.0 * attempt)

            if last_err is not None:
                print(f"❌ id={rid} failed after {args.retries} retries. Stop.", flush=True)
                break

    print("✔ Refinement finished.")

if __name__ == "__main__":
    main()

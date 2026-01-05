import json
import argparse
from pathlib import Path
from utils.llm_client import get_client
from utils.robust_json import try_parse_json
import time


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--provider", type=str, default="groq")
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile")

    # ⭐ 新增：critique 模式
    parser.add_argument(
        "--mode",
        choices=["rubric", "self"],
        default="rubric",
        help="critique generation mode"
    )

    args = parser.parse_args()

    client = get_client(args.provider)

    RESP_PATH = Path("data/responses_llama3_70b.jsonl")
    RUBRIC_PATH = Path("data/final_rubrics_refined_15.jsonl")   # final_rubrics_refined_15.jsonl   rubrics_GPT4o_converted.jsonl   reference_rubrics_converted.jsonl
    PROMPT = Path("prompts/critique_prompt.txt").read_text()
    OUT_FILE = Path("outputs/70b/critique_70b.jsonl")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    responses = load_jsonl(RESP_PATH)

    # ---------- rubric 只在 rubric mode 下加载 ----------
    rubric_by_index = {}
    if args.mode == "rubric":
        rubrics = load_jsonl(RUBRIC_PATH)
        for r in rubrics:
            if "index" in r:
                rubric_by_index[int(r["index"])] = r
            else:
                raise ValueError("Rubric file missing 'index' field")

    # ---------- 断点恢复 ----------
    done_ids = set()
    if OUT_FILE.exists():
        with open(OUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                done_ids.add(int(obj["id"]))

    end = args.end if args.end is not None else len(responses)

    with open(OUT_FILE, "a", encoding="utf-8") as fout:
        for row_i in range(args.start, min(end, len(responses))):
            resp_item = responses[row_i]

            if "id" not in resp_item:
                print(f"⚠️ Skip row {row_i}: response missing id")
                continue

            rid = int(resp_item["id"])
            if rid in done_ids:
                continue

            question = resp_item.get("query", "")
            answer = resp_item.get("answer", "")

            # ---------- 构造 user message ----------
            if args.mode == "rubric":
                rubric_item = rubric_by_index.get(rid)
                if rubric_item is None:
                    print(f"⚠️ Skip id {rid}: no matching rubric")
                    continue

                rubric_payload = rubric_item.get("rubric", rubric_item)

                user_msg = f"""
User question:
{question}

Draft response:
{answer}

Rubric:
{json.dumps(rubric_payload, ensure_ascii=False, indent=2)}
""".strip()

            else:
                # ⭐ Self-critique：不使用 rubric
                user_msg = f"""
User question:
{question}

Draft response:
{answer}

Please carefully review the draft response above.
Identify its main weaknesses, missing or incorrect information,
and propose concrete, actionable suggestions for improvement.
""".strip()

            try:
                llm_resp = client.chat.completions.create(
                    model=args.model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": PROMPT},
                        {"role": "user", "content": user_msg}
                    ]
                )

                raw = llm_resp.choices[0].message.content
                critique, how = try_parse_json(raw)

                if critique is None:
                    debug_dir = Path("outputs/debug_raw")
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    (debug_dir / f"critique_fail_id{rid}_row{row_i}.txt").write_text(
                        raw or "", encoding="utf-8"
                    )
                    raise ValueError(f"Invalid JSON from model (parse={how})")

                fout.write(json.dumps({
                    "id": rid,
                    "row_index": row_i,
                    "critique": critique,
                    "mode": args.mode   # ⭐ 记录 mode，方便后续分析
                }, ensure_ascii=False) + "\n")
                fout.flush()

            except Exception as e:
                print(f"❌ Failed at row {row_i} (id={rid}): {e}")
                break


if __name__ == "__main__":
    main()

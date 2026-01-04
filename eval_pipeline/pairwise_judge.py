from judge_utils import call_judge, safe_parse_json

def build_prompt(rubric_text, question, A, B):
    # 处理不同模式下的基础指令
    if rubric_text and len(rubric_text.strip()) > 0:
        header = f"Compare Response A vs Response B using the provided rubric.\n\n[RUBRIC]\n{rubric_text}"
        item_rule = "items[].id must match rubric numbering (1..N)"
    else:
        # None 模式下的指令
        header = "Compare Response A vs Response B based on overall clinical accuracy, safety, and communication quality."
        item_rule = "Provide a single item in 'items' summarizing the overall comparison."

    return f"""
You are a strict clinical evaluator. {header}

You MUST explicitly check the quality of each response against the required clinical standards.

[QUESTION]
{question}

[RESPONSE A]
{A}

[RESPONSE B]
{B}

Return JSON ONLY in this exact schema:
{{
  "decision": "A|B|SAME",
  "total": {{"A": <number>, "B": <number>, "delta": <number>}},
  "items": [
    {{"id": <int>, "axis": "<string>", "points": <number>, "hit_A": <true/false>, "hit_B": <true/false>}}
  ]
}}

Rules:
- {item_rule}
- Sum item points to compute totals
- decision: A if delta>0, B if delta<0, SAME if very close
- No extra text.
""".strip()

def judge_once(rubric_text, question, A, B):
    prompt = build_prompt(rubric_text, question, A, B)
    raw = call_judge(prompt)
    parsed = safe_parse_json(raw)
    return parsed, raw
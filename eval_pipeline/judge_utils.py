import time
import json
from openai import OpenAI
from config import BASE_URL, OPENAI_API_KEY, JUDGE_MODEL, TEMPERATURE, TIMEOUT

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
    timeout=TIMEOUT,
)

def call_judge(prompt, max_retry=3):
    for attempt in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
            )
            raw = resp.choices[0].message.content.strip()
            return raw
        except Exception as e:
            print(f"[Retry {attempt+1}] Judge API error: {e}")
            time.sleep(2 + attempt)

    raise RuntimeError("Judge failed after retries")


def safe_parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None

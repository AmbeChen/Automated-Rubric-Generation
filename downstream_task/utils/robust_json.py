import json
import re
from typing import Any, Optional, Tuple

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_FIRST_OBJ_RE = re.compile(r"(\{.*\})", re.DOTALL)

def try_parse_json(text: str) -> Tuple[Optional[Any], str]:
    """
    Try parse JSON object from model output.
    Returns (obj_or_None, reason).
    """
    if text is None:
        return None, "text is None"
    s = text.strip()
    if not s:
        return None, "empty response"

    # 1) direct parse
    try:
        return json.loads(s), "direct"
    except Exception:
        pass

    # 2) fenced json block
    m = _JSON_FENCE_RE.search(s)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate), "fence"
        except Exception:
            pass

    # 3) fallback: take first {...} span (greedy), then try
    m = _FIRST_OBJ_RE.search(s)
    if m:
        candidate = m.group(1).strip()
        # sometimes there are multiple objects; try to trim from ends progressively
        try:
            return json.loads(candidate), "brace-span"
        except Exception:
            pass

    return None, "no valid json found"



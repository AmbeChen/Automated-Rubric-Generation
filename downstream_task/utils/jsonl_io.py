import json
from typing import Dict, Any, Optional, Iterable, Tuple, List

CANDIDATE_KEYS = ["id", "index", "conv_id", "qid", "sample_id", "uid", "idx"]

def detect_key(obj: Dict[str, Any]) -> Optional[str]:
    """Return the first existing key name among candidates."""
    for k in CANDIDATE_KEYS:
        if k in obj:
            return k
    return None

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def index_by_key(rows: List[Dict[str, Any]], preferred_key: Optional[str] = None) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    Build {key_value(str): row} map.
    If preferred_key is None, auto-detect from first row.
    """
    if not rows:
        raise ValueError("Empty jsonl")
    key = preferred_key or detect_key(rows[0])
    if not key:
        raise ValueError(f"Cannot find an id key. Tried: {CANDIDATE_KEYS}")
    m = {}
    for r in rows:
        if key not in r:
            # skip rows missing key
            continue
        m[str(r[key])] = r
    return key, m

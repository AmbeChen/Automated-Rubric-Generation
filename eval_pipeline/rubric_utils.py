import json

def load_rubrics_by_index(path):
    rubrics = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            idx = obj.get("index")
            if idx is None:
                continue

            items = obj.get("rubrics")
            if not isinstance(items, list):
                items = obj.get("generated_rubrics")

            if isinstance(items, list):
                rubrics[idx] = items

    return rubrics


def format_rubrics(items):
    lines = []
    lines.append(
        "Apply each rule independently. "
        "Add points when the behavior is present; subtract points when undesirable behavior is present."
    )
    for i, it in enumerate(items, 1):
        crit = str(it.get("criterion", "")).strip()
        axis = str(it.get("axis", "overall")).strip()
        pts = float(it.get("points", 0))
        sign = "+" if pts >= 0 else ""
        lines.append(f"{i}. ({sign}{pts} points, axis={axis}) {crit}")
    return "\n".join(lines)

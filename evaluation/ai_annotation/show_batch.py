import argparse
import json
import re
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("start", type=int, help="zero-based start row")
parser.add_argument("count", type=int)
args = parser.parse_args()

path = Path(__file__).resolve().parent / "blinded_manifest.jsonl"
rows = [json.loads(line) for line in path.open(encoding="utf-8")]
for row in rows[args.start : args.start + args.count]:
    output = row["output"]
    hits = [
        kw
        for kw in row["target_keywords"]
        if re.search(r"(?i)(?<![A-Za-z])" + re.escape(kw) + r"(?![A-Za-z])", output)
    ]
    print(
        f"{row['item_id']} {row['feature_code']} s={row['strength']} "
        f"p={row['prompt_idx']} TARGET={','.join(row['target_keywords'])} "
        f"HIT={','.join(hits)}"
    )
    print("PROMPT:", row["prompt"])
    print("OUT:", output)
    print()

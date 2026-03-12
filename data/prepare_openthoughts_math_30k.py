from datasets import load_dataset
import json

ds = load_dataset("siyanzhao/Openthoughts_math_30k_opsd", split="train")

with open("data/openthoughts_math_30k.jsonl", "w", encoding="utf-8") as f:
    for row in ds:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
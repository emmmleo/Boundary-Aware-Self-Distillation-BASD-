import json
from pathlib import Path


class JsonlLogger:
    def __init__(self, out_file: str):
        self.path = Path(out_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: dict):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

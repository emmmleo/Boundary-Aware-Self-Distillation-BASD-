import json
from pathlib import Path

from torch.utils.data import Dataset

from basd.types import Example


class JsonlMathDataset(Dataset):
    def __init__(self, path: str, max_samples: int | None = None):
        self.path = Path(path)
        self.items: list[Example] = []
        with self.path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_samples is not None and len(self.items) >= max_samples:
                    break
                row = json.loads(line)
                sample_id = row.get("sample_id") or row.get("id") or f"sample-{idx}"
                self.items.append(
                    Example(
                        sample_id=sample_id,
                        question=row["question"],
                        gold_answer=row.get("gold_answer", row.get("answer", "")),
                        reference_solution=row.get("reference_solution", row.get("solution", "")),
                    )
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Example:
        return self.items[idx]

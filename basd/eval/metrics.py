def accuracy(preds: list[bool]) -> float:
    if not preds:
        return 0.0
    return sum(1 for x in preds if x) / len(preds)

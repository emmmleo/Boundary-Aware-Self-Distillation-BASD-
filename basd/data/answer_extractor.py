import re
from fractions import Fraction


def extract_final_answer(text: str) -> str:
    final_match = re.search(r"<<FINAL>>([\s\S]*)$", text)
    if final_match:
        segment = final_match.group(1)
    else:
        tag_match = re.search(r"(?:Final\s*Answer\s*:?)([\s\S]*)$", text, flags=re.IGNORECASE)
        segment = tag_match.group(1) if tag_match else text

    box_match = re.findall(r"\\boxed\{([^}]*)\}", segment)
    if box_match:
        return box_match[-1].strip()

    line = segment.strip().splitlines()[-1] if segment.strip() else ""
    return line.strip()


def normalize_math_answer(ans: str) -> str:
    s = ans.strip().lower().replace(",", "")
    s = re.sub(r"\s+", "", s)
    s = s.replace("$", "")
    if not s:
        return s

    try:
        if "/" in s and all(x for x in s.split("/")):
            return str(float(Fraction(s)))
        return str(float(s))
    except ValueError:
        return s


def is_correct_answer(pred: str, gold: str) -> bool:
    return normalize_math_answer(pred) == normalize_math_answer(gold)

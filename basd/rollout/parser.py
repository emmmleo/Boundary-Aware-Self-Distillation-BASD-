import re

from basd.types import StepSpan


STEP_RE = re.compile(r"<<STEP_(\d+)>>")
FINAL_RE = re.compile(r"<<FINAL>>")


def parse_steps_from_text(text: str) -> list[StepSpan]:
    tags = []
    for m in STEP_RE.finditer(text):
        tags.append((m.start(), m.end(), int(m.group(1)), False))
    for m in FINAL_RE.finditer(text):
        tags.append((m.start(), m.end(), -1, True))
    tags.sort(key=lambda x: x[0])

    spans: list[StepSpan] = []
    for i, (start, end, step_id, is_final) in enumerate(tags):
        content_start = end
        content_end = tags[i + 1][0] if i + 1 < len(tags) else len(text)
        spans.append(
            StepSpan(
                step_id=step_id if not is_final else 10**6,
                char_start=content_start,
                char_end=content_end,
                text=text[content_start:content_end],
                is_final=is_final,
            )
        )
    return spans


def extract_final_region(text: str) -> tuple[int, int]:
    m = FINAL_RE.search(text)
    if not m:
        return len(text), len(text)
    return m.end(), len(text)

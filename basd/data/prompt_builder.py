from dataclasses import dataclass


@dataclass
class PromptConfig:
    step_tag_format: str = "<<STEP_{i}>>"
    final_tag: str = "<<FINAL>>"


def build_student_prompt(question: str, cfg: PromptConfig) -> str:
    example = "\n".join(
        [
            "You are solving a math problem.",
            "Please reason step by step.",
            "Use the exact format:",
            cfg.step_tag_format.format(i=1),
            "...",
            cfg.step_tag_format.format(i=2),
            "...",
            cfg.final_tag,
            "...",
            "",
            "Question:",
            question,
        ]
    )
    return example


def build_teacher_prompt(question: str, reference_solution: str) -> str:
    return "\n".join(
        [
            "You are given a math problem and a verified reference solution.",
            "Use the reference solution only as privileged information for judging",
            "the next token on a student-generated solution trajectory.",
            "",
            "Question:",
            question,
            "",
            "Verified Reference Solution:",
            reference_solution,
            "",
            "Now score the following student reasoning continuation token by token.",
        ]
    )

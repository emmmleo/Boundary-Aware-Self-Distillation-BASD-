from dataclasses import dataclass


STUDENT_SYSTEM_PROMPT = (
    "You are a careful mathematical reasoning assistant. "
    "Reply only in the exact format `Step 1: ...`, `Step 2: ...`, ..., `Final Answer: ...`. "
    "Do not use XML tags, role names, or extra sections."
)

TEACHER_SYSTEM_PROMPT = (
    "You are a privileged teacher model. "
    "You know the verified reference solution. "
    "Your job is to score the student's next token under the same answer prefix, not to rewrite the solution."
)


@dataclass
class PromptConfig:
    max_steps: int = 8


def _render_chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant: "


def build_student_prompt(tokenizer, question: str, cfg: PromptConfig) -> str:
    user_prompt = "\n".join(
        [
            "Solve the problem step by step.",
            "Follow this format exactly:",
            "Step 1: ...",
            "Step 2: ...",
            "...",
            "Final Answer: ...",
            "",
            f"Keep the reasoning concise, use at most {cfg.max_steps} reasoning steps, and stop immediately after `Final Answer:`.",
            "",
            "Problem:",
            question,
        ]
    )
    return _render_chat_prompt(tokenizer, STUDENT_SYSTEM_PROMPT, user_prompt)


def build_teacher_prompt(tokenizer, question: str, reference_solution: str, gold_answer: str) -> str:
    user_prompt = "\n".join(
        [
            "Question:",
            question,
            "",
            "Verified Reference Solution:",
            reference_solution,
            "",
            "Gold Final Answer:",
            gold_answer,
            "",
            "Now score the following student reasoning continuation token by token.",
        ]
    )
    return _render_chat_prompt(tokenizer, TEACHER_SYSTEM_PROMPT, user_prompt)

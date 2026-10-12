"""Dataset loading and task-specific prompt formatting."""

import random

from .constants import (
    CODE_PREFIX,
    DEEPSEEK_CODE_PREFIX,
    GPQA_QUERY_TEMPLATE,
    MATH_COT_SUFFIX,
    MATH_PREFIX,
)

def load_data(dataset_name, split, file_path=None):

    from datasets import load_dataset

    if dataset_name == "MATH":
        return load_dataset("nlile/hendrycks-MATH-benchmark", split=split)
    if dataset_name == "GPQA":
        return load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
    if dataset_name == "HUMANEVAL":
        return load_dataset("openai/openai_humaneval", split=split)
    raise ValueError(f"Unknown dataset: {dataset_name}")

def _user_chat(tokenizer, content):
    """Use the checkpoint's own instruction wrapper, with no added system text."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )

def process_item(item, dataset_name, model_alias, tokenizer):
    """Format a task prompt and return its reference answer and metadata."""
    deepseek = model_alias.startswith("deepseek")
    if not (model_alias.startswith("qwen") or deepseek):
        raise ValueError("Supported model families: Qwen and DeepSeek")

    if dataset_name == "MATH":
        question = item.get("problem") or item.get("prompt")
        solution = item.get("answer") or item.get("solution")
        text = ("" if deepseek else MATH_PREFIX) + question + MATH_COT_SUFFIX
        if deepseek or model_alias in {"qwen_math_grpo", "qwen_grpo"}:
            text = _user_chat(tokenizer, text)
        return text, solution, {"question": question}

    if dataset_name == "GPQA":
        choices = [
            item.get(f"Incorrect Answer {i}", item.get(f"incorrect_answer_{i}"))
            for i in range(1, 4)
        ]
        random.shuffle(choices)
        gold_index = random.randint(0, 3)
        choices.insert(gold_index, item.get("Correct Answer", item.get("correct_answer")))
        question = item.get("Question", item.get("question"))
        query = GPQA_QUERY_TEMPLATE.format(Question=question, **dict(zip("ABCD", choices)))
        text = _user_chat(tokenizer, query) if deepseek else query
        return text, "ABCD"[gold_index], {"question": query, "choices": choices}

    if dataset_name == "HUMANEVAL":
        text = (DEEPSEEK_CODE_PREFIX if deepseek else CODE_PREFIX) + item["prompt"]
        if deepseek:
            text = _user_chat(tokenizer, text)
        return text, item.get("canonical_solution", ""), {
            "task_id": item["task_id"], "prompt": item["prompt"]
        }

    raise ValueError(f"Unknown dataset: {dataset_name}")

import argparse
import ast
import json
import re
import os
import tempfile
from pathlib import Path

from utils.constants import MATH_PREFIX

from utils.grader_utils.math_grader import grade_answer as grade_math
from utils.grader_utils.gpqa_grader import grade_answer as grade_gpqa, parse_answer_gpqa
from utils.grader_utils.he_grader import entry_point as grade_humaneval
from utils.grader_utils.he_check import read_problems

def extract_all_boxed_inners(text: str):
    if not text: return []
    results = []
    for match in re.finditer(r'\\(?:boxed|fbox){', text):
        start = match.end()
        count = 1
        i = start
        while count > 0 and i < len(text):
            if text[i] == "{": count += 1
            elif text[i] == "}": count -= 1
            i += 1
        if count == 0:
            results.append(text[start:i - 1].strip())
    results.reverse()
    return results

MATH_EXPRESSION_PATTERN = re.compile(
    r"\$\$(.*?)\$\$"
    r"|(?<![\\$])\$(?!\$)([^$\n]+?)(?<!\\)\$(?!\$)"
    r"|\\\((.*?)\\\)"
    r"|\\\[(.*?)\\\]",
    flags=re.DOTALL,
)


def math_response_text(text: str):
    text = re.split(r"<\|(?:endoftext|im_end)\|>", text, maxsplit=1)[0]
    for match in re.finditer(r"(?m)^[ \t]*" + re.escape(MATH_PREFIX.rstrip()), text):
        prefix = text[:match.start()]
        if any(extract_all_boxed_inners(prefix)) or re.search(
            r"<answer>\s*\S.*?</answer>", prefix, re.DOTALL | re.IGNORECASE
        ):
            return prefix
    return text


def extract_math_answer(text: str):
    text = math_response_text(text)
    boxed = [answer for answer in extract_all_boxed_inners(text) if answer]
    if boxed:
        return boxed[0]

    conclusions = list(re.finditer(
        r"\b(?:answer\s+is|(?:therefore|thus|so),[^\n.!?]*?\bis)\s+"
        r"([-+]?\d+(?:\.\d+)?(?:/\d+)?)\s*(?:\.(?!\d)|$)",
        text, re.IGNORECASE,
    ))
    candidates = []
    for match in re.finditer(
        r"(?:answer|equal to|is)\s+[:\s]?\s*([0-9a-zA-Z_^{}\\]+)(?:\.|$)",
        text, re.IGNORECASE,
    ):
        candidates.append((match.end(), match.group(1)))
    for match in MATH_EXPRESSION_PATTERN.finditer(text):
        answer = next(group.strip() for group in match.groups() if group is not None)
        if answer:
            candidates.append((match.end(), answer.rsplit("=", 1)[-1].strip()))
    if conclusions:
        conclusion = conclusions[-1]
        later_text = text[conclusion.end():]
        if not re.search(r"\b(?:answer|therefore|thus|so)\b", later_text, re.IGNORECASE):
            return conclusion.group(1)
    if candidates:
        return max(candidates, key=lambda candidate: candidate[0])[1]

    return ""

def evaluate_math(input_file):
    total = 0
    correct = 0

    print(f"Evaluating MATH: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            gold = data.get("solution") or data.get("answer") or data.get("gold_answer")
            completion = data.get("completion", "")

            pred = extract_math_answer(completion)
            if grade_math(pred, gold):
                correct += 1
            total += 1

    print(f"MATH Accuracy: {correct}/{total} = {correct/total:.2%}")

def evaluate_gpqa(input_file):
    total = 0
    correct = 0

    print(f"Evaluating GPQA: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            gold = data.get("solution") or data.get("correct_answer")
            completion = data.get("completion", "")

            try:
                parsed_pred = parse_answer_gpqa(completion)
                if grade_gpqa(parsed_pred, gold):
                    correct += 1
            except Exception:
                pass

            total += 1

    print(f"GPQA Accuracy: {correct}/{total} = {correct/total:.2%}")

def func_name_from_prompt(prompt: str) -> str:
    functions = [
        node for node in ast.parse(prompt or "").body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return functions[-1].name if functions else ""


def _parse_prefix(code: str):
    lines = code.splitlines(keepends=True)
    ends = [len(lines)] + [
        i for i in range(len(lines) - 1, 0, -1)
        if lines[i].strip() and not lines[i].startswith((" ", "\t"))
    ]
    for end in ends:
        prefix = "".join(lines[:end])
        try:
            return prefix, ast.parse(prefix)
        except SyntaxError:
            continue
    return None


def _defined_names(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {alias.asname or (alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name)
                for alias in node.names}
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return {name.id for target in targets for name in ast.walk(target)
                if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)}
    return set()


def _has_implementation(function):
    return any(
        not isinstance(node, ast.Pass) and not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and (isinstance(node.value.value, str) or node.value.value is Ellipsis)
        )
        for node in function.body
    )


def _remove_example_tests(program: str, entry_point: str) -> str:
    tree = ast.parse(program)
    definitions = {}
    for node in tree.body:
        for name in _defined_names(node):
            definitions.setdefault(name, []).append(node)

    required = set()
    pending = [entry_point]
    while pending:
        name = pending.pop()
        if name in required:
            continue
        required.add(name)
        for node in definitions.get(name, []):
            pending.extend(
                child.id for child in ast.walk(node)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
            )

    test_functions = {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name not in required
        and any(isinstance(child, ast.Name) and child.id == entry_point
                for child in ast.walk(node))
        and any(isinstance(child, ast.Assert) for child in ast.walk(node))
    }
    body = [
        node for node in tree.body
        if not isinstance(node, ast.Assert)
        and not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in test_functions
        )
    ]
    if len(body) == len(tree.body):
        return program
    tree.body = body
    return ast.unparse(tree) + "\n"


def extract_code(prompt: str, completion: str, entry_point: str = None) -> str:
    prompt = prompt or ""
    completion = re.split(r"<\|(?:endoftext|im_end)\|>", completion or "", maxsplit=1)[0]
    fn = entry_point or func_name_from_prompt(prompt)
    context = prompt
    for node in ast.parse(prompt).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            context = "".join(prompt.splitlines(keepends=True)[:start - 1])
            break

    fragments = [completion]
    for match in re.finditer(r"(?m)^([ \t]*)(?:\w+:[ \t]*)?```(?:python|py)?[ \t]*\r?\n", completion, re.IGNORECASE):
        indent = match.group(1)
        fragments.append("".join(
            line[len(indent):] if line.startswith(indent) else line
            for line in completion[match.end():].splitlines(keepends=True)
        ))

    candidates = []
    definitions = {}
    for fragment in fragments:
        parsed = _parse_prefix(fragment)
        full_definition = False
        if parsed:
            code, tree = parsed
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not _has_implementation(node):
                    continue
                for name in _defined_names(node):
                    definitions.setdefault(name, node)
            full_definition = any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn
                for node in tree.body
            )
        assembled = (context + code) if full_definition else (prompt.rstrip("\n") + "\n" + fragment)
        parsed = _parse_prefix(assembled)
        if parsed:
            program, tree = parsed
            targets = [node for node in tree.body
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn]
            if targets and _has_implementation(targets[-1]):
                candidates.append((not full_definition, program, tree))

    if not candidates:
        return prompt.rstrip("\n") + "\n" + completion
    _, program, tree = min(candidates, key=lambda candidate: candidate[0])
    defined = {name for node in tree.body for name in _defined_names(node)}
    pending = list(tree.body)
    dependencies = []
    while pending:
        node = pending.pop()
        for child in ast.walk(node):
            if not isinstance(child, ast.Name) or not isinstance(child.ctx, ast.Load):
                continue
            if child.id in defined or child.id not in definitions:
                continue
            dependency = definitions[child.id]
            defined.update(_defined_names(dependency))
            dependencies.append(dependency)
            pending.append(dependency)
    if dependencies:
        ordered = [node for node in definitions.values() if node in dependencies]
        ordered = list({id(node): node for node in ordered}.values())
        index = next(i for i, node in enumerate(tree.body)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn)
        tree.body[index:index] = ordered
        program = ast.unparse(tree) + "\n"
    return _remove_example_tests(program, fn)


def evaluate_humaneval(input_file, problem_file):
    print(f"Preparing HumanEval for execution: {input_file}")

    canonical_data = read_problems(problem_file)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)

                task_id = data.get("task_id", "")
                completion = data.get("completion", "")

                problem = canonical_data[task_id]
                full_code = extract_code(
                    problem["prompt"], completion, entry_point=problem["entry_point"]
                )

                tmp.write(json.dumps({"task_id": task_id, "completion": full_code, "completion_is_full_program": True}, ensure_ascii=False) + "\n")

        tmp_path = tmp.name

    print(f"Running execution (this may take time)...")
    try:
        grade_humaneval(tmp_path, problem_file=problem_file)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["MATH", "GPQA", "HUMANEVAL"])
    parser.add_argument("--input", type=str, required=True, help="Path to input .jsonl file")
    parser.add_argument(
        "--problem-file",
        default=str(Path(__file__).resolve().parent / "data" / "HumanEval.jsonl"),
        help="HumanEval problem JSONL (default: bundled data/HumanEval.jsonl)",
    )
    args = parser.parse_args()
    if args.dataset == "MATH":
        evaluate_math(args.input)
    elif args.dataset == "GPQA":
        evaluate_gpqa(args.input)
    elif args.dataset == "HUMANEVAL":
        evaluate_humaneval(args.input, args.problem_file)

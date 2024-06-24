from __future__ import annotations

import os
import pickle
import re
import string
import subprocess as sp
import typing
import warnings
from collections import Counter
from enum import Enum
from functools import partial
from importlib.util import find_spec
from typing import Callable, Optional, Any, List, Dict, Tuple

from omegaconf import DictConfig

from agent.parsers.parser import ParsingError
from agent.utils import pylogger
from agent.utils import rich_utils

log = pylogger.get_pylogger(__name__)


def break_word_split(break_word: str, raw_response: str):
    parsed_response = raw_response.split(break_word + ": ")[-1]
    return parsed_response


def extract_python(raw_response: str) -> str:
    """Extracts python code from a raw response"""
    python_elements = re.findall("```python([\s\S]*?)```", raw_response)
    if len(python_elements) == 0:
        try:
            python_elements = re.findall("```([\s\S]*?)```", raw_response)
            assert len(python_elements) == 1
            return python_elements[0].replace('python', '')
        except Exception:
            raise ParsingError(
                f"No match found in raw response:\n{raw_response}",
                raw_response,
                f"Did you forget to write 'python' at the beginning of the block in your response?"
            )
    if len(python_elements) > 1:
        raise ParsingError(
            f"Too many matches in {raw_response}",
            raw_response,
            "Did you return more than one python code blocks? Please return only one."
        )
    try:
        return python_elements[0]
    except Exception as e:
        print("Raw:\n", raw_response)
        raise ParsingError(f"Failed to parse", raw_response, e.args[0]) from e


def extract_json(raw_response: str) -> Dict[str, Any]:
    """Extracts a returned json object from a raw response"""
    json_elements = re.findall("```json([\s\S]*?)```", raw_response)
    if len(json_elements) == 0:
        try:
            return eval(raw_response)
        except Exception:
            raise ParsingError(
                f"No match found in raw response:\n{raw_response}",
                raw_response,
                f"Did you forget to write 'json' at the beginning of the block in your response?"
            )
    if len(json_elements) > 1:
        raise ParsingError(
            f"Too many matches in {raw_response}",
            raw_response,
            "Did you return more than one json code blocks? Please return only one."
        )
    try:
        return eval(json_elements[0])
    except Exception as e:
        print("Raw:\n", raw_response)
        raise ParsingError(f"Failed to run eval({json_elements[0]})", raw_response, e.args[0]) from e


def extract_json_with_bools(raw_response: str) -> Dict[str, bool]:
    raw_response = raw_response.replace(": true", ": True")
    raw_response = raw_response.replace(": false", ": False")
    response = extract_json(raw_response)
    for k, v in response.items():
        if v not in [True, False]:
            raise ParsingError(
                f"The value associated to key `{k}` is of type `{type(v)}` but it should be of type `Boolean`."
            )
    return response


def extract_as_json(raw_response: str, matchname: Optional[str]) -> str:
    """Catch the result of a response given in json style"""
    json_elements = re.findall("```json([\s\S]*?)```", raw_response)
    if len(json_elements) == 0:
        try:
            candidate = raw_response.replace("\n", "")
            candidate = re.sub("[\"'\{\}]", "", candidate)
            k, v = candidate.split(":")
            if matchname is not None:
                assert k.strip().lower() == matchname.lower()
            return v.strip()
        except Exception:
            raise ParsingError(
                f"No match found in raw response:\n{raw_response}",
                raw_response,
                f"Did you forget to write 'json' at the beginning of the block in your response?"
            )
    if len(json_elements) > 1:
        raise ParsingError(
            f"Too many matches in {raw_response}",
            raw_response,
            "Did you return more than one json code blocks? Please return only one."
        )
    try:
        return eval(json_elements[0])[matchname]
    except (SyntaxError, TypeError):
        candidate = json_elements[0].replace("\n", "")
        candidate = re.sub("[\"'\{\}]", "", candidate)
        k, v = candidate.split(":")
        if matchname is not None:
            assert k.strip().lower() == matchname.lower()
        return v.strip()
    except Exception as e:
        print("Raw:\n", raw_response)
        raise ParsingError(f"Failed to extract key '{matchname}' from {json_elements[0]}",
                           raw_response, e.args[0]) from e


extract_action_as_json = partial(extract_as_json, matchname="action")
extract_command_as_json = partial(extract_as_json, matchname="command")
extract_summary_as_json = partial(extract_as_json, matchname="summary")
extract_submission_as_json = partial(extract_as_json, matchname="submission")
extract_metric_as_json = partial(extract_as_json, matchname="metric")


def concat_files(
        path_file1: str,
        path_file2: str,
        out_path: str,
        map_content1: Optional[Callable[[str], str]] = None,
        map_content2: Optional[Callable[[str], str]] = None,
) -> None:
    """Adding the contents of two files to merge them together, and save as out_path file."""
    with open(path_file1) as f:
        contents_file1 = f.readlines()
        if map_content1:
            contents_file1 = map_content1("\n".join(contents_file1)).split("\n")
    with open(path_file2) as f:
        contents_file2 = f.readlines()
        if map_content2:
            contents_file2 = map_content2("".join(contents_file2)).split("\n")
    out_content = contents_file1 + contents_file2
    with open(out_path, "w") as f:
        f.writelines(out_content)


def save_w_pickle(obj: Any, path: str, filename: Optional[str] = None) -> None:
    """Save object obj in file exp_path/filename.pkl"""
    if filename is None:
        filename = os.path.basename(path)
        path = os.path.dirname(path)
    if len(filename) < 4 or filename[-4:] != ".pkl":
        filename += ".pkl"
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, filename), "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
    - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
    - save the exception to a `.log` file
    - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
    - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[dict, dict]:

        ...

        return metric_dict, object_dict
    ```
    """

    def wrap(cfg: DictConfig):
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def normalize_answer(answer: str) -> str:
    """normalize the sentence."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(answer))))


def pairwise_similarity(answer1: str, answer2: str) -> float:
    """Calculate similarity between two answers."""

    normalized_answer1 = normalize_answer(answer1)
    normalized_answer2 = normalize_answer(answer2)

    answer1_tokens = normalized_answer1.split()
    answer2_tokens = normalized_answer2.split()
    common = Counter(answer1_tokens) & Counter(answer2_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(answer1_tokens)
    recall = 1.0 * num_same / len(answer2_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def check_code_safety(code: str) -> None:
    """Checks code for unsafe commands such as 'rm' or 'mv' and others that can compromise data on the machine."""
    pattern_list = [
        "rm ",
        "mv ",
        "cp ",
        "chmod",
        "sudo",
        "mkdir",
        "wget",
        "curl",
        "zip",
        "unzip",
        "pip",
        "conda",
        "rmdir",
        "apt-get",
    ]
    for pattern in pattern_list:
        if ("os.system()" in code or "sp.Popen" in code) and pattern in code:
            raise ValueError(f"Unsafe command '{pattern}' found in code! Aborting.")


class ListableEnum(Enum):
    @classmethod
    def list(cls) -> List[Any]:
        return list(map(lambda c: c.value, cls))

    @classmethod
    def rev_dict(cls) -> Dict[Any, Any]:
        return {c.value: c for c in cls if isinstance(c.value, typing.Hashable)}


class HumanInputCancelError(Exception):
    pass


def human_input(allow_cancel: bool, w=150) -> str:
    if os.getenv("NO_HUMAN", False):
        print("Queried human input in NO_HUMAN mode!")
        exit(1)
    stop_signal = "[STOP]"
    redo_flag = "[REDO]"
    cancel_flag = "[CANCEL]"

    lines = ["-" * w]
    text = ("Rules for manual input in terminal:\n"
            "- write as many lines as you want\n"
            f"- when you're done, put stop flag: {stop_signal}\n"
            f"- if you spot a mistake in a previous line and you want to retry, enter : {redo_flag}")
    if allow_cancel:
        text += f"\n- if you want to cancel manual input, enter cancel flag : {cancel_flag}"
    text_lines = justify_text(text=text, w=w - 4)
    text_lines = list(map(lambda t: f"| {t} |", text_lines))
    lines.extend(text_lines)
    lines.append("-" * w)

    print("\n".join(lines))

    done = False
    n_trial = 0
    while not done:
        reply = ""
        output = input(f"\n===> Please {'re' if n_trial > 0 else ''}start your reply: ")
        if allow_cancel and cancel_flag in output:
            raise HumanInputCancelError()
        n_empty = 0
        while stop_signal not in output and redo_flag not in output:
            reply += output + "\n"
            output = input("")
            if allow_cancel and cancel_flag in output:
                raise HumanInputCancelError()
            if output == "":
                n_empty += 1
            else:
                n_empty = 0
            if n_empty > 5:
                print(f"When you're done, put stop flag: {stop_signal}")
        if redo_flag in output:
            continue

        output = output[:len(output) - len(stop_signal)]
        reply += output
        done = True
    print("===> End of edition mode, resume running <===")
    return reply


def str_justify_one_line(s: str, w: int) -> List[str]:
    """
    Given a string containing a single-line input, return a list of strings of fixed length to allow single
    -column display
    """
    if s == "":
        return [" " * w]
    lines = []
    last_char = " "
    i = 0
    while i < len(s):
        new_line = ""
        if last_char.isalnum() and s[i].isalnum():
            new_line += "-"
        end = i + w - len(new_line)
        new_line += s[i:end]
        new_line += " " * (w - len(new_line))
        lines.append(new_line)
        i = end
        last_char = new_line[-1]
    return lines


def justify_text(text: str, w: int) -> List[str]:
    """ Given a text, get a list of strings such that the text can be printed in a column of fixed length"""
    all_lines = text.split("\n")
    justified_text = []
    for line in all_lines:
        aux_line = line.replace("｜", "|").replace("\t", "    ")
        justified_text.extend(str_justify_one_line(s=aux_line, w=w))
    return justified_text


def print_in_two_cols(t1: str, t2: str, w: int = 88) -> None:
    """ Given two texts, print them in parallel in two separate columns"""
    w = (w - 7) // 2
    col1 = justify_text(t1, w=w)
    col2 = justify_text(t2, w=w)
    max_lines = max(len(col1), len(col2))
    col1 = [" " * w for _ in range(max_lines - len(col1))] + col1
    col2 = [" " * w for _ in range(max_lines - len(col2))] + col2

    print("┌" + "─" * (w + 2) + "┬" + "─" * (w + 2) + "┐")
    for i in range(max_lines):
        print("│ " + col1[i] + " │ " + col2[i] + " │")
    print("└" + "─" * (w + 2) + "┴" + "─" * (w + 2) + "┘")


def run_cmd(command: str) -> Tuple[str, str]:
    r = sp.Popen([command], stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    out, err = r.communicate()
    out, err = out.decode("ascii").strip(), err.decode()
    return out, err


def run_python_code(
        workspace_path: str,
        path_to_python: str,
        code_path: str,
        code_output_path: str,
        code_warnings_path: str,
        code_error_path: str,
        aux_code_error_path: str
) -> Tuple[str, str, str]:
    """
    Runs a python script and catches errors.
    This function also catches errors related to the script not running (aux error).

    Args:
        workspace_path: (str) the path to the current workspace
        path_to_python: (str) the path to the python executable used to run the code
        code_path: (str) the path to the python code to be run
        code_output_path: (str) the path to the output file
        code_warnings_path: (str) the path to the warnings file
        code_error_path: (str) the path to the error file
        aux_code_error_path: (str) the path to the aux error file

    Returns:
        a tuple containing the code output and code error

    """
    cmd = f"cd {workspace_path} && {path_to_python} {code_path} 2> {aux_code_error_path} > {code_output_path}"
    os.system(cmd)

    # Catch errors --> find some errors in aux_code_error_path, then any error in code_error_path
    with open(aux_code_error_path) as f:
        aux_code_error = f.readlines()
        if len(aux_code_error) > 0:
            aux_code_error = aux_code_error[-1]
            errors_to_catch = ["SyntaxError:", "TabError:", "IndentationError:"]
            caught_error = False
            for error in errors_to_catch:
                if error in aux_code_error:
                    code_error = aux_code_error
                    caught_error = True
                    break
            if not caught_error:
                if os.path.exists(code_error_path):
                    with open(code_error_path) as ff:
                        code_error = ff.read()
                    os.remove(code_error_path)
                else:
                    code_error = ""
        else:
            code_error = ""

    with open(code_output_path) as f:
        code_output = f.read()

    if code_error == "":
        with open(aux_code_error_path) as f:
            code_error = f.read()

    code_warnings = ""
    if code_error == "" and os.path.exists(code_warnings_path):
        with open(code_warnings_path) as f:
            code_warnings = f.read()

    if os.path.exists(code_output_path):
        os.remove(code_output_path)
    if os.path.exists(code_warnings_path):
        os.remove(code_warnings_path)
    if os.path.exists(code_error_path):
        os.remove(code_error_path)
    if os.path.exists(aux_code_error_path):
        os.remove(aux_code_error_path)

    return code_output, code_warnings, code_error


def catch_error_wrap(code: str, code_error_path: str, code_warnings_path: str) -> str:
    """ Wrap the code in a try/except to catch any error raised while running the code and write it into a file. """
    wrapped_code = "import os\n"
    wrapped_code += "import traceback\n"
    wrapped_code += "import warnings\n"
    wrapped_code += "\n"
    wrapped_code += "def write_warning_to_file(message, category, filename, lineno, file=None, line=None):\n"
    wrapped_code += f"    with open('{code_warnings_path}', 'w') as f:\n"
    wrapped_code += ("        f.write(warnings.formatwarning(message=message, category=category, filename=filename, "
                     "lineno=lineno, line=line))\n")
    wrapped_code += "\n"
    wrapped_code += "warnings.showwarning = write_warning_to_file\n"
    wrapped_code += "\n"
    wrapped_code += "try:\n"
    wrapped_code += "\n".join(map(lambda x: "    " + x, code.split("\n")))
    wrapped_code += "\nexcept Exception as e:\n"
    wrapped_code += "    error_message = traceback.format_exc()\n"
    wrapped_code += f"    with open('{code_error_path}', 'w') as f:\n"
    wrapped_code += "        f.write(error_message)\n"
    wrapped_code += "        raise e\n"
    return wrapped_code


def get_path_to_python(path_to_python: str) -> str:
    while not os.path.exists(path_to_python):
        python_exe = input(
            rf"/!\ file {os.path.abspath(path_to_python)} should contain the absolute path to the python"
            f" executable to use, but {path_to_python} does not exists...\n"
            f"Enter here the path to the python executable you want to use:\n"
        )
        os.system(f'echo "{python_exe}" > {path_to_python}')
    with open(path_to_python) as f:
        python_exe = f.readline().replace("\n", "")
    return python_exe

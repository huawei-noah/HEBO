from __future__ import annotations

import ast
import os
import pickle
import re
import string
import subprocess as sp
import time
import traceback
import typing
import warnings
from collections import Counter
from enum import Enum
from functools import partial
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, Optional, Any, Tuple

import pandas as pd
from omegaconf import DictConfig

from agent.parsers.parser import ParsingError, UnsupportedExtensionError
from agent.utils import pylogger
from agent.utils import rich_utils

log = pylogger.get_pylogger(__name__)
SUPPORTED_FILE_EXTENSIONS = ['.txt', '.csv', '.tsv', '.json']


def get_agent_root_dir() -> Path:
    """ Read and return the root path where agent code is located """
    return Path(__file__).parent.parent.parent.parent


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


def extract_json(raw_response: str) -> dict[str, Any]:
    """Extracts a returned json object from a raw response"""
    json_elements = re.findall("```json([\s\S]*?)```", raw_response)

    if len(json_elements) == 0:
        try:
            return eval(raw_response)
        except Exception:
            raise ParsingError(
                f"No match found in raw response:\n{raw_response}",
                raw_response,
                f"Did you forget to write 'json' at the beginning of the block in your response?\n"
                f"Did you forget to write your response in a ```json...``` block?"
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


def extract_json_with_bools(raw_response: str) -> dict[str, bool]:
    raw_response = raw_response.replace(": true", ": True")
    raw_response = raw_response.replace(": false", ": False")
    response = extract_json(raw_response)
    for k, v in response.items():
        if v not in [True, False]:
            raise ParsingError(
                f"The value associated to key `{k}` is of type `{type(v)}` but it should be of type `Boolean`."
            )
    return response


def extract_as_json(raw_response: str, matchname: str | None) -> str:
    """Catch the result of a response given in json style"""

    def get_val_from_dict_str(candidate_) -> str:
        candidate_ = ast.literal_eval(candidate_)
        if matchname is not None:
            assert list(candidate_.keys())[0].lower() == matchname.lower()
        if isinstance(list(candidate_.values())[0], str):
            return list(candidate_.values())[0].strip()
        return list(candidate_.values())[0]

    # check if there are nested JSON structures
    peeled_raw_response = raw_response[raw_response.find("```json") + 7:raw_response.rfind("```")].strip()
    if len(re.findall(pattern="```json([\s\S]*?)```", string=peeled_raw_response)) > 0:
        # If there are multiple ```json delimiters, this could mean nested json structures
        start_delim = "```json"
        end_delim = "```"
        pattern = (rf'({re.escape(start_delim)}((?:[^{re.escape(start_delim)}]|'
                   rf'{re.escape(start_delim)}(?!{re.escape(end_delim)}))*)'
                   rf'(?:{re.escape(end_delim)}(?!.*{re.escape(start_delim)}.*{re.escape(end_delim)})))')
        match = re.search(pattern=pattern, string=raw_response, flags=re.DOTALL)
        if match:
            return get_val_from_dict_str(candidate_=match.group(1))
        else:
            raise ParsingError(
                f"Nested JSON structures detected in:\n{raw_response}",
                raw_response,
                f"Your answer should not contain nested JSON structures. Use the desired format strictly."
            )
    else:
        json_elements = re.findall(pattern="```json([\s\S]*?)```", string=raw_response)

    if len(json_elements) == 0:
        try:
            dict_elements = re.findall("{([\s\S]*?)}", raw_response)
            if len(dict_elements) > 0:
                candidate = "{" + dict_elements[0].replace("\n", "") + "}"
            else:
                if matchname in raw_response:
                    candidate = "{" + raw_response.replace("\n", "") + "}"
                else:
                    candidate = "{'" + matchname + "': '" + raw_response.replace("\n", "") + "'}"
            return get_val_from_dict_str(candidate_=candidate)
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
        candidate = json_elements[0].replace("\n", "")
        return get_val_from_dict_str(candidate_=candidate)
    except Exception as e:
        print("Raw:\n", raw_response)
        raise ParsingError(f"Failed to extract key '{matchname}' from {json_elements[0]}",
                           raw_response, e.args[0]) from e


def extract_paths_as_json(raw_response: str) -> str:
    paths = extract_as_json(raw_response=raw_response, matchname="paths")
    # check if all files exist in response list
    correct_paths = []
    invalid_paths = []
    unsupported_extension_paths = []
    dir_paths = []
    for path in paths:
        if not os.path.exists(path):
            invalid_paths.append(path)
        elif not os.path.splitext(path)[-1] in SUPPORTED_FILE_EXTENSIONS:
            unsupported_extension_paths.append(path)
        elif os.path.isdir(path):
            dir_paths.append(path)
        else:
            correct_paths.append(path)

    if len(invalid_paths) > 0:
        if len(invalid_paths) > 1:
            err_str = '\n\t- ' + '\n\t- '.join([path for path in invalid_paths])
        else:
            err_str = f'\n\t- {invalid_paths[0]}'
        raise FileNotFoundError(f"The following paths do not exist:" + err_str, raw_response)

    if len(unsupported_extension_paths) > 0:
        if len(unsupported_extension_paths) > 1:
            err_str = '\n\t- '.join([path for path in unsupported_extension_paths])
        else:
            err_str = f'\n\t- {unsupported_extension_paths[0]}'
        raise UnsupportedExtensionError(
            f"Only attempt to read files with extension in [{SUPPORTED_FILE_EXTENSIONS}].\n"
            f"The following files have unsupported extensions:" + err_str, raw_response
        )

    if len(dir_paths) > 0:
        if len(dir_paths) > 1:
            err_str = '\n\t- '.join([path for path in dir_paths])
        else:
            err_str = f'\n\t- {dir_paths[0]}'
        raise FileNotFoundError(f"The following paths are not files but directories:" + err_str, raw_response)

    return paths


def extract_detected_file_as_json(raw_response: str) -> str | None:
    detected_file = extract_as_json(raw_response=raw_response, matchname="detected_file")
    # check if file exists
    if detected_file is None or "no file" in detected_file.lower():
        return None
    elif not os.path.exists(detected_file):
        raise FileNotFoundError(f"The detected file {detected_file} do not exist.", raw_response)
    elif not os.path.splitext(detected_file)[-1] in SUPPORTED_FILE_EXTENSIONS:
        raise UnsupportedExtensionError(
            f"Only attempt to read files with extension in [{SUPPORTED_FILE_EXTENSIONS}].\n"
            f"The detected file {detected_file} has unsupported extension {os.path.splitext(detected_file)[-1]}",
            raw_response
        )
    elif os.path.isdir(detected_file):
        raise FileNotFoundError(f"The detected file path {detected_file} does not point to a file but to a directory",
                                raw_response)
    else:
        return detected_file


def extract_column_names_and_values_as_json(raw_response: str, path_to_df: str) -> dict[str, Any]:
    df = pd.read_csv(path_to_df)
    name_val_dict = extract_json(raw_response=raw_response)
    for name, val in name_val_dict.items():
        if name not in df.columns:
            raise ParsingError(
                f"Column name {name} not in target column names!",
                raw_response,
                f"Only use as keys of the returned JSON, the names in the list provided."
            )
        if val not in set(df[name]):
            raise ParsingError(
                f"Positive class name {val} not in the values of target column {name}!",
                raw_response,
                f"Only use as values of the return JSON, the class labels of the target column {name}."
            )
    return name_val_dict


extract_action_as_json = partial(extract_as_json, matchname="action")
extract_command_as_json = partial(extract_as_json, matchname="command")
extract_summary_as_json = partial(extract_as_json, matchname="summary")
extract_submission_as_json = partial(extract_as_json, matchname="submission")
extract_metric_as_json = partial(extract_as_json, matchname="metric")
extract_instruction_as_json = partial(extract_as_json, matchname="instruction")
extract_hyperparams_as_json = partial(extract_as_json, matchname="hyperparams")


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


def save_w_pickle(obj: Any, path: str, filename: str | None = None, overwrite: bool = True) -> None:
    """Save object obj in file exp_path/filename.pkl
    Args:
        overwrite: whether to overwrite existing file
    """
    if filename is None:
        filename = os.path.basename(path)
        path = os.path.dirname(path)
    if len(filename) < 4 or filename[-4:] != ".pkl":
        filename += ".pkl"
    if not os.path.exists(path):
        os.makedirs(path)
        os.chmod(path, 0o777)
    filepath = os.path.join(path, filename)
    if not os.path.exists(filepath) or overwrite:
        with open(filepath, "wb") as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        os.chmod(filepath, 0o777)


def load_w_pickle(path: str, filename: str | None = None) -> Any:
    """ Load object from file exp_path/filename.pkl """
    if filename is None:
        filename = os.path.basename(path)
        path = os.path.dirname(path)
    if len(filename) < 4 or filename[-4:] != '.pkl':
        filename += '.pkl'
    p = os.path.join(path, filename)
    with open(p, 'rb') as f:
        try:
            return pickle.load(f)
        except EOFError:
            raise Exception(f"EOFError with {p}")
        except UnicodeDecodeError:
            raise Exception(f"UnicodeDecodeError with {p}")
        except pickle.UnpicklingError:
            raise Exception(f"UnpicklingError with {p}")


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
    def list(cls) -> list[Any]:
        return list(map(lambda c: c.value, cls))

    @classmethod
    def rev_dict(cls) -> dict[Any, Any]:
        return {c.value: c for c in cls if isinstance(c.value, typing.Hashable)}


class HumanInputCancelError(Exception):
    pass


def human_input(allow_cancel: bool, w=150) -> str:
    if os.getenv("NO_HUMAN", False):
        traceback.print_stack()
        print("Queried human input in NO_HUMAN mode!")
        exit(1)
    stop_signal = "[STOP]"
    redo_flag = "[REDO]"
    cancel_flag = "[CANCEL]"

    lines = ["─" * w]
    text = ("Rules for manual input in terminal:\n"
            "- write as many lines as you want\n"
            f"- when you're done, put stop flag: {stop_signal}\n"
            f"- if you spot a mistake in a previous line and you want to retry, enter : {redo_flag}")
    if allow_cancel:
        text += f"\n- if you want to cancel manual input, enter cancel flag : {cancel_flag}"
    text_lines = justify_text(text=text, w=w - 4)
    text_lines = list(map(lambda t: f"| {t} |", text_lines))
    lines.extend(text_lines)
    lines.append("─" * w)

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


def str_justify_one_line(s: str, w: int) -> list[str]:
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


def justify_text(text: str, w: int) -> list[str]:
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


def unwrap_code(wrapped_code: str) -> str:
    """
    Finds the code inside a try/except and returns it indented left
    """
    inner_code = re.findall("try:([\s\S]*?)except", wrapped_code)[0]
    return inner_code.replace("\n    ", "\n").strip()


def get_path_to_python(path_to_python: str) -> str:
    while not os.path.exists(path_to_python):
        os.makedirs(os.path.dirname(path_to_python), exist_ok=True)
        python_exe = input(
            rf"/!\ file {os.path.abspath(path_to_python)} should contain the absolute path to the python"
            f" executable to use, but {path_to_python} does not exists...\n"
            f"Enter here the path to the python executable you want to use:\n"
        )
        os.system(f'echo "{python_exe}" > {path_to_python}')
    with open(path_to_python) as f:
        python_exe = f.readline().replace("\n", "")
    return python_exe


def time_formatter(t: float, show_ms: bool = False) -> str:
    """ Convert a duration in seconds to a str `dd:hh:mm:ss`

    Args:
        t: time in seconds
        show_ms: whether to show ms on top of dd:hh:mm:ss
    """
    n_day = time.gmtime(t).tm_yday - 1
    if n_day > 0:
        ts = time.strftime('%H:%M:%S', time.gmtime(t))
        ts = f"{n_day}:{ts}"
    else:
        ts = time.strftime('%H:%M:%S', time.gmtime(t))
    if show_ms:
        ts += f'{t - int(t):.3f}'.replace('0.', '.')
    return ts


if __name__ == "__main__":
    example = 'sad\n{"summary": "it;sa\'as sdfas \nsadasfd asd "\n}'
    assert extract_as_json(raw_response=example, matchname="summary")
    example = '\n{"summary": "it;sa\'as sdfas \nsadasfd asd "\n}'
    assert extract_as_json(raw_response=example, matchname="summary")
    example = '\n```{"summary": "it;sa\'as sdfas \nsadasfd asd "\n}```'
    assert extract_as_json(raw_response=example, matchname="summary")
    example = '\n```json{"summary": "it;sa\'as sdfas \nsadasfd asd "\n}```'
    assert extract_as_json(raw_response=example, matchname="summary")

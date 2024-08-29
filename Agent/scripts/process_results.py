import math
import os
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import yaml
from rich import print
from rich.console import Console
from rich.theme import Theme

console = Console(theme=Theme({"warning": "bold red"}))

warnings.filterwarnings("ignore")

_EXCLUDE_IF_ERRORS = {
    "alfworld": 5,
    "gsm8k": 1000,  # we have expected index out of range error
    "hotpotqa": 150,
    "webshop": 5,
    "humaneval": 5,
    "babyai": 10,
}

_LLM_FORMAT_DICT = {
    "gpt3.5": "GPT-3.5",
    "openchat": "OpenChat",
    "llama": "Llama-13b",
    "mistral": "Mistral",
    "vicuna13b": "Vicuna-13b",
    "llama70b": "Llama-70b",
}

_METHOD_FORMAT_DICT = {
    "direct": "Direct",
    "zs-cot": "ZS-CoT",
    "fs": "FS",
    "fs-cot": "FS-CoT",
    "fs-cot-sc": "FS-CoT-SC",
    "fs-cot-react": "FS-CoT-React",
    "fs-cot-reflect": "FS-CoT-Reflect",
    "fs-cot-zerostep-reflect": "FS-CoT-Reflect",
    "fs-least2most": "FS-Least-to-Most",
    "fs-cot-swift-sage": "FS-CoT-Swift-Sage",
}

_SINGLE_STEP_TASKS = {
    "gsm8k",
    "hotpotqa",
    "humaneval",
}


def print_latex_table_to_file(results, filename):
    with open(filename, "w") as file:
        headers = ["Method", "Overall", "ALFWorld", "GSM8K", "HotpotQA", "Webshop", "Humaneval", "BabyAI"]
        formatted_headers = [f"{h: <25}" for h in headers]
        file.write(f"& {headers[0]:<26} & " + " & ".join(formatted_headers[1:]) + r"\\")
        file.write("\n\\midrule \n")
        for llm in _LLM_FORMAT_DICT.keys():
            # Header formatting
            file.write(f"% LLM: {llm.capitalize()}\n")
            file.write(
                "\\multirow{4}{*}{\\rotatebox[origin=c]{90}"
                + f"{{\\scriptsize{{\\textbf{{{_LLM_FORMAT_DICT[llm]}}}}}}}"
                + "}\n"
            )

            for method in [
                "direct",
                "zs-cot",
                "fs",
                "fs-cot",
                "fs-cot-sc",
                "fs-cot-react",
                "fs-cot-reflect",
                "fs-cot-swift-sage",
                "fs-least2most",
            ]:
                row = []
                overall = []
                for task in ["alfworld", "gsm8k", "hotpotqa", "webshop", "humaneval", "babyai"]:
                    # Special case for single-step tasks reflection
                    current_method = method
                    if current_method == "fs-cot-reflect" and task in _SINGLE_STEP_TASKS:
                        current_method = "fs-cot-zerostep-reflect"

                    key = (f"llm@_agents.0.llm={llm}", f"method={current_method}", f"task={task}")
                    print_key = f"'( LLM= {llm:<9}, method= {current_method:<23}, task= {task:<9} )'"

                    # If no results for this method, task, and LLM, add dash or N/A
                    if key not in results:
                        # If the method is not applicable to this task, add a dash
                        cell = (
                            "-"
                            if (current_method == "fs-cot-swift-sage" and task in ["gsm8k", "hotpotqa", "humaneval"])
                            or (current_method == "fs-least2most" and task in ["alfworld", "webshop", "babyai"])
                            or (current_method == "fs-cot-sc" and task in ["humaneval"])
                            else "N/A"
                        )
                        row.append(cell)
                        continue

                    # Get avg rewards as percentages, and filter out runs with too many errors (except gsm8k)
                    values = results[key]
                    filtered_avgs = [
                        value["reward_avg"] * 100
                        for value in values
                        if value["reward_fails"] <= _EXCLUDE_IF_ERRORS[task]
                    ]
                    if len(filtered_avgs) < len(values):
                        errors = np.sum([value["reward_fails"] for value in values])
                        print(
                            f"{print_key}| filtered out {len(values) - len(filtered_avgs)}/{len(values)} runs, "
                            f"total errors: {errors}"
                        )

                    # Warn if less than 3 successful runs
                    if len(filtered_avgs) < 3:
                        console.print(
                            f"{print_key}| WARNING: only {len(filtered_avgs)} successful runs found",
                            style="warning",
                        )

                    # Calculate mean and std across seeds
                    mean = np.mean(filtered_avgs)
                    std = np.std(filtered_avgs)

                    overall.append(mean)
                    cell = f"{mean:.1f} \\small{{$ \\pm \\ {std:.1f}$}}" if not math.isnan(mean) else "N/A"
                    row.append(cell)

                # Add overall average and add formatted row to table
                overall_avg = np.nanmean(overall)
                row.insert(0, f"\\textbf{{{overall_avg:.1f}}}") if not math.isnan(overall_avg) else row.insert(0, "N/A")
                formatted_row = [f"{r: <25}" for r in row]
                file.write(
                    f"& \\textbf{{{_METHOD_FORMAT_DICT[current_method]: <17}}} & {' & '.join(formatted_row)} \\\\\n"
                )

            file.write("\\midrule \n\n")


def main(path: Path):
    # get all subdirectories of path
    runs = [
        dir for dir, subdirs, _ in os.walk(path) if ".hydra" in subdirs
    ]  # if not dir.endswith(".hydra") and "llm" in subdir]
    print(f"Found {len(runs)} runs...")
    results = defaultdict(list)

    for run in runs:
        try:
            data = pd.read_json(path_or_buf=f"{run}/output.jsonl", lines=True)

            with open(f"{run}/.hydra/config.yaml") as file:
                config = yaml.safe_load(file)

            with open(f"{run}/.hydra/overrides.yaml") as file:
                overrides = yaml.safe_load(file)

            num_episodes = data["episode"].nunique()
            assert config["max_episodes"] == num_episodes

            num_fails = len(data[data["reward"] == "fail"])
            data = data[data["reward"] != "fail"]

            dp = {
                "episodes": num_episodes,
                "reward_avg": data["reward"].sum() / num_episodes,
                "reward_fails": num_fails,
            }

            # for o in overrides:
            #     tag = "llm@_agents.0.llm="
            #     if tag in o:
            #         llm = o.split("=")[1]
            #         if llm == "openchat2":
            #             o = o.replace("openchat2", "openchat")
            overrides = tuple(
                sorted(
                    [
                        o.replace("openchat2", "openchat")
                        for o in overrides
                        if any(s in o for s in ["task=", "method=", "llm@_agents.0.llm="])
                    ]
                )
            )
            results[overrides].append(dp)

        except KeyboardInterrupt:
            break
        except Exception:
            run = run[len(str(path)) + 1 :]
            print(f"Skipping {run} (unfinished or broken)")
            continue

    print("-" * 140)
    console.print("RESULTS", style="bold")
    print_latex_table_to_file(results, "results.tex")
    print("-" * 140)


if __name__ == "__main__":
    typer.run(main)

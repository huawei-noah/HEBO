import gc
import os
import re
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch
import typer
from peft import PeftModel
from rich import print
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from typing_extensions import Annotated

running_version = None


def get_local_ip():
    try:
        # This creates a dummy socket to connect to a public DNS server.
        # It's used only to find out the local IP address.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"  # Fallback to localhost in case of failure


# Extracts version number from model name
def get_version(name):
    version = re.search(r"-(\d+)$", name)
    return int(version.group(1)) if version else -1


# Monitor the model folder for the latest version
def get_latest_model_path(models_directory):
    """Returns the path to the latest model in the given directory.

    Args:
        models_directory (str): Path to the directory containing the models.
    """
    # TODO: make this cleaner?
    folders = [
        f
        for f in os.listdir(models_directory)
        if os.path.isdir(os.path.join(models_directory, f)) and "checkpoint" in f and "tmp" not in f
    ]
    if not folders:
        return None
    latest = max(folders, key=get_version)
    return latest


def build_commands(cfg, kwargs):
    """Builds the commands to start the worker processes.

    Args:
        cfg (AttrDict): Config object
        kwargs (dict): Additional arguments to pass to the worker process
    """
    num_processes = len(cfg.gpus)
    unique_env_vars = [os.environ.copy() for i in range(num_processes)]

    for i, env_vars in enumerate(unique_env_vars):
        env_vars.update({"CUDA_VISIBLE_DEVICES": f"{cfg.gpus[i]}"})
        env_vars.update({"LOGDIR": tempfile.TemporaryDirectory().name})

    unique_process_args = [
        {
            "port": f"{cfg.base_port + cfg.gpus[i]}",
            "worker-address": f"http://{cfg.base_worker_address}:{cfg.base_port + cfg.gpus[i]}",
            **cfg.kwargs,
            **kwargs,
        }
        for i in range(num_processes)
    ]
    commands = []
    for envs, args in zip(unique_env_vars, unique_process_args):
        command = f"{cfg.base_worker_script_command}"
        for key, val in args.items():
            if val != "null":
                command += f" --{key}={val}"
            else:
                command += f" --{key}"
        commands.append((command, envs))
    return commands


def start_processes(cfg):
    """Starts the worker processes and returns a list of processes.

    Args:
        cfg (AttrDict): Config object
    """

    if cfg.lora_dir and cfg.auto_load_lora:
        # find latest checkpoint from lora_dir
        latest = get_latest_model_path(cfg.lora_dir)
    elif cfg.lora_dir:
        # ask user for latest checkpoint. As options, give him all folders in lora_dir
        print("Please select a model to load from lora_dir")
        folders = [f for f in os.listdir(cfg.lora_dir) if os.path.isdir(os.path.join(cfg.lora_dir, f))]
        print(folders)
        latest = input("Please input the folder name: ")
    else:
        latest = None

    if latest:
        merge_base_and_lora(cfg, latest)
        gc.collect()
        time.sleep(1)
        with torch.cuda.device(f"cuda:{cfg.gpus[0]}"):
            torch.cuda.empty_cache()
        time.sleep(2)

    launch_kwargs = {"model-path": f"{cfg.merged_save_dir}/{latest}"} if latest else {}

    commands = build_commands(cfg, launch_kwargs)
    processes = []
    for i, (command, env_vars) in enumerate(commands):
        print(f"Starting worker process with command: {command}")
        file = Path(f"vllm_logs/vllm_worker_{i}.out")
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, "w") as outfile:
            process = subprocess.Popen(command.split(" "), env=env_vars, stdout=outfile, stderr=outfile)
        processes.append(process)
    print(f"{len(commands)} Processes have been started!")
    return processes


def merge_base_and_lora(cfg, latest):
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id, torch_dtype=torch.float16, device_map=f"cuda:{cfg.gpus[0]}"
    )
    lora_path = cfg.lora_dir + "/" + latest
    print(f"Loading lora weights from {lora_path}...")
    model = PeftModel.from_pretrained(base_model, lora_path, device_map=f"cuda:{cfg.gpus[0]}")

    save_path = cfg.merged_save_dir + f"/{latest}"
    print(f"Merging and saving to {save_path}...")
    merged = model.merge_and_unload()
    merged.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Merging and saving completed")


def monitor_and_restart_processes(cfg, processes):
    global running_version

    # search for new version...
    if cfg.lora_dir and cfg.auto_load_lora:
        latest = get_latest_model_path(cfg.lora_dir)
    else:
        latest = None

    launch_kwargs = {"model-path": f"{cfg.merged_save_dir}/{latest}"} if latest else {}

    if latest == running_version:
        commands = build_commands(cfg, launch_kwargs)
        for (i, process), command in zip(enumerate(processes), commands):
            if process.poll() is not None:  # If the process is dead
                print(f"Restarting dead worker process ID: {i}...")
                print(f"Starting worker process with command: {command[0]}")
                file = Path(f"vllm_logs/vllm_worker_{i}.out")
                file.parent.mkdir(parents=True, exist_ok=True)
                with open(file, "w") as outfile:
                    new_process = subprocess.Popen(
                        command[0].split(" "), env=command[1], stdout=outfile, stderr=outfile
                    )
                processes[i] = new_process
        return

    # new version has been detected.
    # we will first terminate the first process (first gpu)
    if processes[0].poll() is None:
        print(f"Terminating process on gpu {cfg.gpus[0]} in order to merge model")
        print(f"Terminating process {processes[0]}")
        processes[0].send_signal(signal.SIGTERM)
    processes[0].wait()
    # then merge base with lora and save into a new path
    merge_base_and_lora(cfg, latest)

    # cleanup the first gpu
    time.sleep(1)
    gc.collect()
    time.sleep(20)
    with torch.cuda.device(f"cuda:{cfg.gpus[0]}"):
        torch.cuda.empty_cache()
    time.sleep(20)
    running_version = latest

    commands = build_commands(cfg, launch_kwargs)

    for (i, process), command in zip(enumerate(processes), commands):
        print(f"Restarting process ID: {i}...")
        if process.poll() is None:
            process.send_signal(signal.SIGINT)  # Gracefully terminate the process
        process.wait()  # Wait for the process to terminate
        print(f"Starting worker process with command: {command[0]}")
        file = Path(f"vllm_logs/vllm_worker_{i}.out")
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, "w") as outfile:
            new_process = subprocess.Popen(command[0].split(" "), env=command[1], stdout=outfile, stderr=outfile)
        time.sleep(60)  # stagger the process creation so we are not left without online models
        processes[i] = new_process
    print("Worker processes have been restarted if needed!")


class AttrDict(dict):
    """Dictionary subclass that allows attribute access."""

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


def main(
    gpus: Annotated[
        str, typer.Option(help="Comma-separated list of integers representing GPU IDs to use.", show_default=False)
    ],
    base_model_id: Annotated[str, typer.Option(help="ID of the base model to be used.", show_default=False)],
    base_worker_script_command: Annotated[
        str, typer.Option(help="Base command to start the worker script.")
    ] = "python3 -m fastchat.serve.vllm_worker",
    base_port: Annotated[
        int, typer.Option(help="Base port number for the vLLM workers. Actual port is {base_port} + {gpu_id}")
    ] = 31000,
    base_worker_address: Annotated[
        str, typer.Option(help="Worker's address required for the controller. Defaults to the machine's local IP.")
    ] = get_local_ip(),
    auto_load_lora: Annotated[bool, typer.Option(help="Flag to automatically load LoRA.")] = False,
    lora_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Directory of LoRA weights. Required if using auto-load-lora.",
            file_okay=False,
            dir_okay=True,
            exists=True,
        ),
    ] = None,
    merged_save_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Directory to save merged cache. Required if using auto-load-lora or lora-dir.",
            file_okay=False,
            dir_okay=True,
            writable=True,
            exists=True,
        ),
    ] = None,
    # Following arguments are forwarded to vllm workers
    model_names: Annotated[
        Optional[str],
        typer.Option(
            help="[kwargs] Names of the model separated by comma. "
            "Defaults to '{base_model_id}-sft-auto' if not specified. Passed to vLLM workers. ",
        ),
    ] = None,
    model_path: Annotated[
        Optional[str],
        typer.Option(
            help="[kwargs] Path to the model. Defaults to {base-model-id} if not specified. "
            "Automatically updated to merged LoRA when necessary. Passed to vLLM workers.",
        ),
    ] = None,
    host: Annotated[str, typer.Option(help="[kwargs] Host address. Passed to vLLM workers.")] = "0.0.0.0",
    controller: Annotated[
        str, typer.Option(help="[kwargs] Controller URL. Passed to vLLM workers.")
    ] = "http://localhost:21001",
    dtype: Annotated[str, typer.Option(help="[kwargs] Data type to be used. Passed to vLLM workers.")] = "float16",
    max_num_batched_tokens: Annotated[
        Optional[int],
        typer.Option(help="[kwargs] Maximum number of batched tokens (recommended!). Passed to vLLM workers."),
    ] = None,
    conv_template: Annotated[
        Optional[str],
        typer.Option(help="[kwargs] Conversation template identifier (recommended!). Passed to vLLM workers."),
    ] = None,
    extra_kwargs: Annotated[
        Optional[str],
        typer.Option(help="[kwargs] Additional key-value pairs for kwargs, formatted as 'key1=value1,key2=value2,...'"),
    ] = None,
):
    """Launches vLLM workers on specified GPUs using a base model, with optional configurations for
    LoRA layers and other parameters.

    Args:
        gpus (str): Comma-separated list of GPU IDs to be used.
        base_model_id (str): Identifier for the base model.
        base_worker_script_command (str): Command to start the worker script.
            Default is 'python3 -m fastchat.serve.vllm_worker'.
        base_port (int): Base port number for vLLM workers. Actual port used is base_port + gpu_id. Default is 31000.
        base_worker_address (str): Address required for the controller, defaulting to the machine's local IP.
        auto_load_lora (bool): Flag indicating whether to automatically load LoRA layers. Default is False.
        lora_dir (Path, optional): Directory containing LoRA weights, required if auto_load_lora is True.
        merged_save_dir (Path, optional): Directory for saving merged cache,
            required if using auto-load-lora or lora-dir.
        model_names (str, optional): Comma-separated names of the models,
            defaults to '{base_model_id}-sft-auto' if not specified.
        model_path (str, optional): Path to the model, defaults to {base_model_id} and updated for LoRA when necessary.
        host (str): Host address for vLLM workers. Default is '0.0.0.0'.
        controller (str): URL of the controller for vLLM workers. Default is 'http://localhost:21001'.
        dtype (str): Data type for processing. Default is 'float16'.
        max_num_batched_tokens (int, optional): Maximum number of batched tokens.
        conv_template (str, optional): Conversation template identifier.
        extra_kwargs (str, optional): Additional key-value pairs for worker configuration,
            formatted as 'key1=value1,key2=value2,...'.

    Returns:
        None: This function does not return a value but launches vLLM worker processes.

    Examples:
        To learn more about this script run:

            $ python scripts/vllm_worker_dispatcher --help

        To launch two workers on each of two GPUs (e.g. 0 and 1) using the base model
            'meta-llama/Llama-2-7b-chat-hf', use the following command:

            $ python scripts/vllm_worker_dispatcher.py --gpus 0,1 --base_model_id meta-llama/Llama-2-7b-chat-hf
    """
    gpu_list = [int(gpu.strip()) for gpu in gpus.split(",")]
    extra_kwargs_dict = {}
    if extra_kwargs:
        extra_kwargs_list = extra_kwargs.split(",")
        for pair in extra_kwargs_list:
            key, value = pair.split("=")
            extra_kwargs_dict[key.strip()] = value.strip()

    if auto_load_lora:
        if not lora_dir or not lora_dir.is_dir():
            typer.echo(
                "Error: --lora-dir must be provided and must be a valid directory if using --auto_load_lora.",
                err=True,
            )
            raise typer.Exit(code=1)
        if not merged_save_dir or not merged_save_dir.is_dir():
            typer.echo(
                "Error: --merged-save-dir must be provided and must be a valid directory if using --auto-load-lora.",
                err=True,
            )
            raise typer.Exit(code=1)

    if lora_dir and merged_save_dir is None:
        typer.echo(
            "Error: --merged-save-dir must be provided and must be a valid directory if using --lora-dir.",
            err=True,
        )
        raise typer.Exit(code=1)

    cfg = AttrDict(
        {
            "gpus": gpu_list,
            "base_worker_script_command": base_worker_script_command,
            "base_port": base_port,
            "base_worker_address": base_worker_address,
            "base_model_id": base_model_id,
            "lora_dir": lora_dir,
            "auto_load_lora": auto_load_lora,
            "merged_save_dir": merged_save_dir,
            "kwargs": {
                "model-names": model_names or f"{base_model_id}-sft-auto",
                "model-path": model_path or base_model_id,
                "host": host,
                "dtype": dtype,
                "controller": controller,
                **extra_kwargs_dict,
            },
        }
    )

    if max_num_batched_tokens:
        cfg.kwargs.update({"max-num-batched-tokens": max_num_batched_tokens})
    if conv_template:
        cfg.kwargs.update({"conv-template": conv_template})

    print(dict(cfg))

    try:
        processes = start_processes(cfg)
        time.sleep(55)
        while True:
            time.sleep(5)
            monitor_and_restart_processes(cfg, processes)

    except KeyboardInterrupt:
        print("Interrupt received! Stopping all processes...")
    except Exception as e:
        print(f"Unexpected error occurred: {e}. Stopping all processes...")
    finally:
        # Send termination signal to all processes
        for process in processes:
            if process.poll() is None:  # If the process is still running
                process.send_signal(signal.SIGINT)

        # Wait for all processes to terminate
        for process in processes:
            process.wait()  # Wait for the process to terminate

        print("All processes have been stopped.")


if __name__ == "__main__":
    typer.run(main)

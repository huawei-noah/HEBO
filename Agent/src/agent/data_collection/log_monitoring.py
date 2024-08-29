import os
import time
import traceback
from pathlib import Path

import ujson
from pathos.multiprocessing import ProcessingPool as Pool
from rich import print

from agent.data_collection.log_processor import LogProcessor
from agent.data_collection.log_processor import check_is_same_method
from agent.data_collection.log_processor import instantiate_components
from agent.data_collection.log_processor import retrieve_and_filter_data


def init_log_processors(
    path: Path,
    use_parsed_output: bool = False,
    new_method: str = None,
    new_task: str = None,
    chat_format: bool = False,
):
    """Initialize log watchers."""
    # Get all relevant log files
    paths = [dir for dir, subdirs, _ in os.walk(path) if ".hydra" in subdirs]

    if not paths:
        raise ValueError(f"No log files found in {path}")

    # Get prompt builder and memory
    prompt_builder, memory = instantiate_components(paths[0], new_method, new_task, chat_format)

    # Process each log file into a data class
    with Pool() as p:
        same_method_list = p.map(lambda path: check_is_same_method(path, new_method), paths)

    log_processors = [
        LogProcessor(p, use_parsed_output, prompt_builder, memory, new_method, is_same_method)
        for p, is_same_method in zip(paths, same_method_list)
    ]

    return log_processors


def process_logs_into_dataset(
    path: Path,
    log_processors: list,
    returns_threshold: float,
    best_episode_dict: dict = {},
    deduplicate_episodes: bool = False,
    deduplicate_timesteps: str = "no",
    chat_format: bool = False,
):
    """Convert logs into a dataset."""
    prev_new_episodes = len(best_episode_dict)

    # Update data
    current_states = [(lp.path, lp.episode_checkpoint) for lp in log_processors]

    with Pool() as p:
        new_data = p.map(lambda s: retrieve_and_filter_data(*s, returns_threshold), current_states)

    for processor, (episode_logs, mem_store_keys, episode_checkpoint) in zip(log_processors, new_data):
        processor.update_data(episode_logs, mem_store_keys, episode_checkpoint)

    # Get shortest episodes for each unique task
    # if deduplicate_episodes:
    for processor in log_processors:
        best_episode_dict.update(processor.get_best_episodes(best_episode_dict))

    total_new_episodes = len(best_episode_dict)
    print(f"New positive unique episodes: {total_new_episodes - prev_new_episodes}")
    print(f"Total positive unique episodes: {total_new_episodes}")

    # best_ids = list(best_episode_dict.keys())
    # # save best ids to json file
    # with open('best_ids.json', 'w') as outfile:
    #     json.dump(best_ids, outfile, indent=4)

    # save best episodes to json file

    with open(path / "current_best_episodes.json", "w") as outfile:
        ujson.dump(best_episode_dict, outfile, indent=4)

    # Create dataset from shortest episodes across all log files
    dataset = []
    best_dict = best_episode_dict if deduplicate_episodes else {}
    for processor in log_processors:
        processor.add_dataset(dataset, best_dict, deduplicate_timesteps, chat_format)

    return dataset


def monitor_logs_for_conversion(
    path: Path,
    output_path: Path = None,
    returns_threshold: float = 0.0001,
    use_parsed_output: bool = False,
    new_method: str = None,
    new_task: str = None,
    interval: int = 30,
    deduplicate_episodes: bool = False,
    deduplicate_timesteps: str = "no",
    chat_format: bool = False,
    filter_initial_episodes: bool = False,
):
    """Start watching logs for new episodes and convert them into a dataset."""
    log_processors = init_log_processors(
        path,
        use_parsed_output,
        new_method,
        new_task,
        chat_format,
    )

    # Check if there are already best episodes
    if filter_initial_episodes and os.path.exists(path / "current_best_episodes.json"):
        with open(path / "current_best_episodes.json") as infile:
            best_episode_dict = ujson.load(infile)

        # Marginally increment timestep in case of old best episodes
        for key in best_episode_dict:
            best_episode_dict[key][1] -= 1
    else:
        best_episode_dict = {}

    print(f"Start unique episodes (not added): {len(best_episode_dict)}")

    # Watch log files for new episodes every <interval> seconds
    iteration = 0
    while True:
        try:
            dataset = process_logs_into_dataset(
                path,
                log_processors,
                returns_threshold,
                best_episode_dict,
                deduplicate_episodes,
                deduplicate_timesteps,
                chat_format,
            )
            print(f"Dataset size: {len(dataset)}")

            # Save dataset to jsonl file
            output_path = path / "collected_data.jsonl" if output_path is None else output_path
            with open(output_path, "a") as outfile:
                for entry in dataset:
                    ujson.dump(entry, outfile)
                    outfile.write("\n")
        except ValueError as e:
            print(str(traceback.format_exc()))
            print(e)
            print("Skipping this iteration...")

        print(f"Sleeping for {interval} seconds...")
        time.sleep(interval)
        iteration += 1

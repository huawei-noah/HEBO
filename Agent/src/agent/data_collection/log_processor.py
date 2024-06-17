import hydra
import numpy as np
import pandas as pd
import yaml
from hydra import compose
from hydra import initialize
from rich import print


def instantiate_components(path: str, new_method: str, new_task: str, chat_format: bool = False):
    """Update config and instantiate prompt builder, LLM, and memory."""
    with open(f"{path}/.hydra/overrides.yaml") as file:
        overrides = yaml.safe_load(file)

    new_overrides = []
    for o in overrides:
        if o.startswith("method="):
            o.split("=")[-1]
            o = f"method={new_method}" if new_method is not None else o
        elif o.startswith("task=") and new_task is not None:
            o.split("=")[-1]
            o = f"task={new_task}"
        elif o.startswith("llm"):
            o = "llm@agent.llm=openchat_3.5"
        new_overrides.append(o)

    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(
            config_name="default_sa_eval.yaml",
            return_hydra_config=True,
            overrides=new_overrides,
        )

    if chat_format:
        if "autoregressive" not in cfg.agent["prompt_builder"]["template_paths"]:
            cfg.agent["prompt_builder"]["template_paths"].insert(0, "autoregressive")
    elif "autoregressive" in cfg.agent["prompt_builder"]["template_paths"]:
        cfg.agent["prompt_builder"]["template_paths"].remove("autoregressive")

    prompt_builder = hydra.utils.instantiate(cfg.agent["prompt_builder"])
    if "_partial_" in cfg.agent["llm"]:
        cfg.agent["llm"]["_partial_"] = False
    llm = hydra.utils.instantiate(cfg.agent["llm"])
    prompt_builder.llm = llm
    memory = hydra.utils.instantiate(cfg.agent["memory"])

    return prompt_builder, memory


def check_is_same_method(path: str, new_method: str):
    with open(f"{path}/.hydra/overrides.yaml") as file:
        overrides = yaml.safe_load(file)
    is_same_method = True
    for o in overrides:
        if o.startswith("method="):
            method = o.split("=")[-1]
            is_same_method = method == new_method or new_method is None
            o = f"method={new_method}" if new_method is not None else o
            print(f"Method: {method} -> {o.split('=')[-1]}")
    return is_same_method


def _process_data_into_episodes(data, episode_nums):
    """Process data into episodes, and return the shortest episode for each unique task."""
    episode_logs = {}
    for e in episode_nums:
        episode_data = data[data["episode"] == e]
        task_id_list = episode_data["task_id"].dropna() if "task_id" in episode_data.columns else []

        # If task_id is not present, use the first observation
        if len(task_id_list) == 0:
            task_id = episode_data["memory:store:observation"].dropna().iloc[0]
        else:
            task_id = task_id_list.iloc[0]

        current_shortest = episode_logs[task_id][0] if task_id in episode_logs else np.inf
        last_timestep = int(episode_data["timestep"].max())
        if last_timestep < current_shortest:
            episode_logs[task_id] = (last_timestep, episode_data)
    return episode_logs


def retrieve_and_filter_data(path, episode_checkpoint, returns_threshold):
    """Read in data and filter out episodes with returns below a certain threshold."""
    data = pd.read_json(path_or_buf=f"{path}/output.jsonl", lines=True)
    episode_nums = data["episode"].unique()

    episode_nums = episode_nums[episode_nums > episode_checkpoint]
    episode_nums = (
        episode_nums[:-1]
        if len(episode_nums) > 0 and data[data["episode"] == episode_nums[-1]]["done"].iloc[0] is np.nan
        else episode_nums
    )
    episode_checkpoint = episode_nums[-1] if len(episode_nums) > 0 else episode_checkpoint

    if len(episode_nums) == 0:
        print("No new episodes found.")
        return {}, [], episode_checkpoint

    data["reward"] = pd.to_numeric(data["reward"], errors="coerce")
    # TODO: filter out with nan reward
    for e in episode_nums:
        v = data[data["episode"] == e]
        returns = v["reward"].sum()
        data.loc[data["episode"] == e, "returns"] = returns

    data = data[data["returns"] >= returns_threshold]
    episode_nums = data["episode"].unique()
    print(f"Found {len(episode_nums)} new episodes.")

    episode_logs = _process_data_into_episodes(data, episode_nums)
    mem_store_keys = [x for x in data.columns if x.startswith("memory:store:")]
    return episode_logs, mem_store_keys, episode_checkpoint


class LogProcessor:
    def __init__(
        self,
        path: str,
        use_parsed_output: bool,
        prompt_builder,
        memory,
        method: str,
        is_same_method: bool,
    ):
        self.path = path
        self.output_key = "llm:parsed_output" if use_parsed_output else "llm:output"
        self.episode_checkpoint = -1
        self.is_same_method = is_same_method
        self.prompt_builder = prompt_builder
        self.memory = memory
        self.method = method

    def update_data(self, episode_logs, mem_store_keys, episode_checkpoint):
        """Update data and episode numbers."""
        self.episode_checkpoint = episode_checkpoint
        self.episode_logs = episode_logs
        self.mem_store_keys = mem_store_keys

    def get_best_episodes(self, best_episode_dict: dict = {}):
        """Get the best episode for each unique task."""
        new_additions = {}
        for task_id, (last_timestep, episode_data) in self.episode_logs.items():
            reward = float(round(episode_data["reward"].sum(), 8))
            if reward > 0:
                if (
                    # If this episode has not yet been seen
                    task_id not in best_episode_dict
                    # If this episode has a higher reward than the previous best
                    or reward > best_episode_dict[task_id][0]
                    # If this episode has the same reward but is shorter than the previous best
                    or (reward == best_episode_dict[task_id][0] and last_timestep < best_episode_dict[task_id][1])
                ):
                    new_additions[task_id] = (reward, last_timestep)
        return new_additions

    def add_dataset(
        self, dataset, best_episode_dict: dict = {}, deduplicate_timesteps: str = "no", chat_format: bool = False
    ):
        """Create dataset from (best, unique - optionally) episodes across all log files."""
        pos_eps = 0
        neg_eps = 0
        for task_id, (last_timestep, episode_data) in self.episode_logs.items():
            reward = round(episode_data["reward"].sum(), 8)
            if reward == 0 or (
                not best_episode_dict
                or (reward == best_episode_dict[task_id][0] and last_timestep == best_episode_dict[task_id][1])
            ):
                if reward > 0:
                    pos_eps += 1
                elif neg_eps >= pos_eps:
                    continue
                else:
                    neg_eps += 1
                    reward = -1
                self._process_episode(
                    episode_data, reward, dataset, task_id, best_episode_dict != {}, deduplicate_timesteps, chat_format
                )

    def _process_episode(
        self, episode_data, reward, dataset, task_id, deduplicate_episodes, deduplicate_timesteps, chat_format
    ):
        """Process a single episode and add it to the dataset."""
        assert deduplicate_timesteps == "no" or not chat_format, "Cannot deduplicate timesteps in chat format."

        self.memory.reset()
        output = None
        token_probs_key = "llm:chosen_token_probs"
        curr_templates = ["external_action.jinja"]

        # Ensure true episode deduplication by checking if the task_id has already been seen
        if deduplicate_episodes:
            for d in dataset:
                if d["task_id"] == task_id:
                    return

        new_msgs = []
        curr_token_probs = {}
        all_token_probs = []
        tmp_dataset = self.prompt_builder(["chat_system_prompt.jinja"], {"memory": self.memory}) if chat_format else {}
        for i, row in episode_data.iterrows():
            # Filter out outputs that are not actions if converting to a different method
            check_filter_actions = self.is_same_method or curr_templates == ["external_action.jinja"]

            # Create dataset sample at every output (or only at actions if converting to a different method)
            if isinstance(row[self.output_key], str) and check_filter_actions:
                # If chat format, add observation to user message
                if chat_format:
                    obs = self.memory.retrieve({"text_obs": 1.0})
                    new_msgs = [{"role": "user", "content": obs}]

                # If not chat format, reset message to system prompt + user prompt with template
                else:
                    new_msgs = self.prompt_builder(curr_templates, {"memory": self.memory})

                # Add output as assistant message
                output = row[self.output_key]

                if self.method == "direct":
                    if "Thought:" in output:
                        # TODO: Improve the way we parse just the action
                        # WARNING: only works for "Action" keywords
                        output = output[output.find("Action:") :]

                new_msgs.append({"role": "assistant", "content": output})

                if chat_format:
                    tmp_dataset.extend(new_msgs)
                    all_token_probs.append(curr_token_probs)
                elif curr_templates != ["external_action.jinja"]:
                    tmp_dataset[i] = tmp_dataset[i] + [new_msgs] if i in tmp_dataset else [new_msgs]

            # Store current template if present
            elif "templates" in row and isinstance(row["templates"], str):
                curr_templates = row["templates"]

            # Store token probabilities if present
            elif token_probs_key in row and isinstance(row[token_probs_key], list):
                curr_token_probs = row[token_probs_key]

            # If current log row is a memory store, store the memory
            else:
                for k in self.mem_store_keys:
                    if isinstance(row[k], str):
                        # If storing an action, also add the entry to the dataset
                        if not chat_format and k.endswith(":action"):
                            tmp_dataset[row[k]] = (
                                tmp_dataset[row[k]] + [new_msgs] if row[k] in tmp_dataset else [new_msgs]
                            )

                        keys = k.split(":")[-1].split("|")
                        if isinstance(keys, str):
                            keys = [keys]
                        self.memory.store(row[k], set(keys))

        if chat_format:
            dataset.append(
                {
                    "task_id": task_id,
                    "messages": tmp_dataset,
                    "token_probs": all_token_probs,
                    "reward": reward,
                }
            )
            return

        # Deduplicate timesteps
        if deduplicate_timesteps == "first":
            tmp_dataset = {action: [all_msgs_for_action[0]] for action, all_msgs_for_action in tmp_dataset.items()}
        elif deduplicate_timesteps == "last":
            tmp_dataset = {action: [all_msgs_for_action[-1]] for action, all_msgs_for_action in tmp_dataset.items()}

        all_entries = [
            {
                "task_id": task_id,
                "messages": msgs,
            }
            for all_msgs_for_action in tmp_dataset.values()
            for msgs in all_msgs_for_action
        ]
        dataset.extend(all_entries)

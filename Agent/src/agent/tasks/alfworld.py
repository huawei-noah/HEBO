import os
import re
from typing import Any, Dict

import alfworld.agents.environment as environment
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu

from agent.memory import MemKey
from agent.tasks import ActionSpace
from agent.tasks import Task
from agent.utils import break_word_split
from agent.utils import pylogger

log = pylogger.get_pylogger(__name__)


class AlfWorld(Task):
    def __init__(self, train_eval, filter_game_files, seed=None, **kwargs):
        super().__init__(**kwargs)

        self.action_space = ActionSpace.DISCRETE

        self.kwargs = kwargs
        self.process_actions = self.kwargs["env"].get("process_actions", False)
        env_type = self.kwargs["env"]["type"]  # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

        # setup environment
        env = getattr(environment, env_type)(self.kwargs, train_eval=train_eval)

        if filter_game_files:
            new_game_files = []
            for gf in env.game_files:
                if any([fgf in gf for fgf in filter_game_files]):
                    new_game_files.append(gf)
            env.game_files = new_game_files

        self.env = env.init_env(batch_size=1)
        if seed is not None:
            self.env.seed(seed)
        self.timesteps = 0
        self.task_prefixes = [
            "look_at_obj",
            "pick_and_place",
            "pick_clean_then_place",
            "pick_cool_then_place",
            "pick_heat_then_place",
            "pick_two_obj",
        ]

    def reset(self, next_subtask: str | None = None) -> Dict[str, Any]:
        """Reset the environment and return the initial observation."""

        if next_subtask:
            base_path = f"{os.environ['ALFWORLD_DATA']}/json_2.1.1/train/"
            next_subtask = base_path + next_subtask + "/game.tw-pddl"
            self.env.unwrapped._gamefiles_iterator = iter([next_subtask])

        obs, info = self.env.reset()
        self.task_category, self.id = self._retrieve_task_info(info["extra.gamefile"])
        return self._return_observation(obs, info)

    # TODO: Improve usability of this. Maybe default implementation,
    # maybe can have an overarching method which covers all cases?
    # and each task can override if necessary to add new functionality
    def answer_parser(self, raw_response: str) -> str:
        return break_word_split("Action", raw_response)

    def step(self, action: str) -> tuple[Dict[str, Any], float, bool]:
        """Perform an action and return the next observation, reward, and done."""

        if self.process_actions:
            action = process_action(action.replace("</s>", "\n"), self.available_actions)
        obs, scores, dones, infos = self.env.step([action])
        return self._return_observation(obs, infos), scores[0], dones[0]

    def _return_observation(self, obs: list[str], infos: Dict[str, Any]) -> Dict[str, Any]:
        """Return the observation dictionary."""

        return {
            MemKey.TASK_CATEGORY: self.task_category,
            MemKey.OBSERVATION: obs[0],
            MemKey.AVAILABLE_ACTIONS: infos["admissible_commands"][0],
        }

    def _retrieve_task_info(self, task_paths: list[str]) -> tuple[str, str]:
        """Retrieve task category and id from the task file."""

        if len(task_paths) > 1:
            raise ValueError("More than one task file provided.")
        task_path_split = task_paths[0].split("/")
        task_filename = task_path_split[-3]
        task_id = ",".join(task_path_split[-4:-1])

        for task_prefix in self.task_prefixes:
            if task_filename.startswith(task_prefix):
                return task_prefix, task_id

        raise ValueError("Task file does not correspond to one of the task categories.")


def bleu_score(reference, candidate):
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()

    smoothie = SmoothingFunction().method4
    score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)
    return score


def process_action(action, choices, limit=0.01, logging=False):
    if logging:
        log.info("Raw action: %s", action)
    match = re.search("ACTION:(.*)\n", action)
    if match:
        action = match.group(1)
    else:
        match = re.search("ACTION:(.*)", action)
        if match:
            action = match.group(1)

    action = action.strip().lower().split("\n")[0]
    if not choices:
        return action
    if action in choices:
        return action
    try:
        bleus = [bleu_score(choice, action) for choice in choices]
        max_index = np.argmax(np.array(bleus))
        max_score = bleus[max_index]
        if max_score > limit:
            if logging:
                log.info("Processed action: %s, score: %f", choices[max_index], max_score)
            return choices[max_index]
    except Exception as e:
        log.exception(e)
        log.debug("choices: %s", choices)
        log.debug("action: %s", action)
    return action

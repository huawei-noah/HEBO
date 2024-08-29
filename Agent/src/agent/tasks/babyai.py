import warnings
from enum import Enum
from typing import Any, Dict

import babyai
import gym

from agent.memory import MemKey
from agent.tasks.tasks import ActionSpace
from agent.tasks.tasks import Task
from agent.utils import break_word_split
from agent.utils import pylogger

log = pylogger.get_pylogger(__name__)

# This log exists to convince ruff that the babyai import is useful
log.info("Imported %s environments", babyai.__name__)


class BabyAI(Task):
    INVALID_ACTION_MESSAGE = "The previous action was invalid, please try a different action. "

    class InvalidActionEffect(str, Enum):
        FEEDBACK = "feedback"
        ERROR = "error"
        REPEAT = "repeat"
        FORWARD = "forward"

    def __init__(
        self,
        env_name,
        invalid_action_effect="feedback",
        env_kwargs=None,
        seed=None,
        action_space: str = "discrete",
        **kwargs,
    ):
        assert "BabyAI" in env_name
        super().__init__(**kwargs)

        env_kwargs = env_kwargs if env_kwargs is not None else {}
        self.env: gym.Env = gym.make(env_name, **env_kwargs)

        self.action_space = ActionSpace.DISCRETE if action_space == "discrete" else ActionSpace.CONTINUOUS

        self.invalid_action_effect = BabyAI.InvalidActionEffect[invalid_action_effect.upper()]
        if seed is not None:
            self.env.seed(seed)

        self.available_actions = {}

    def reset(self, next_subtask: str | None = None) -> Dict[str, Any]:
        """Reset the environment and return the initial observation."""

        if next_subtask is not None:
            warnings.warn("BabyAI does not support subtasks, ignoring subtask")

        obs, info = self.env.reset()
        return self._return_observation(obs, info, first=True)

    def answer_parser(self, raw_response: str) -> str:
        return break_word_split("Action", raw_response)

    def step(self, action: str) -> tuple[Dict[str, Any], float, bool]:
        """Perform an action and return the next observation, reward, and done."""

        action = self.available_actions[action] if action in self.available_actions else ""
        action = 2 if action is None and self.invalid_action_effect is BabyAI.InvalidActionEffect.FORWARD else action

        if action is None:
            if self.invalid_action_effect is BabyAI.InvalidActionEffect.FEEDBACK:
                if not self.text_obs.startswith(self.INVALID_ACTION_MESSAGE):
                    self.text_obs = self.INVALID_ACTION_MESSAGE + self.text_obs
            return self._format_observation(), 0, False

        obs, rew, done, info = self.env.step(action)
        return self._return_observation(obs, info), rew, done

    def _return_observation(self, obs, info, first=False) -> Dict[str, Any]:
        """Return the observation dictionary."""

        text_obs = "What you see ahead of you:\n" + "\n".join("  - " + obs for obs in info["observation"])

        # available_actions_str = '0: rotate left, 1: rotate right, 2: move forward, 3: pickup an object'
        available_actions_str = info["available actions"]
        self.available_actions = {}
        for action in available_actions_str.split(", "):
            action_id, action_desc = action.split(": ")
            self.available_actions[action_desc.lower()] = int(action_id)

        obs_prefix = f"Your task is to: {obs['mission']}. " if first else ""
        self.text_obs = obs_prefix + text_obs
        return self._format_observation()

    def _format_observation(self) -> str:
        """Return the formatted observation."""

        return {
            MemKey.OBSERVATION: self.text_obs,
            MemKey.AVAILABLE_ACTIONS: list(self.available_actions.keys()),
        }

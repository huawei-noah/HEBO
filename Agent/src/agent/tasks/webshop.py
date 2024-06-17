from typing import Any, Dict

from web_agent_site.envs import WebAgentTextEnv
from web_agent_site.utils import DEBUG_PROD_SIZE

from agent.memory import MemKey
from agent.tasks import ActionSpace
from agent.tasks import Task
from agent.utils import break_word_split
from agent.utils import pylogger

log = pylogger.get_pylogger(__name__)


class WebShop(Task):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.action_space = ActionSpace.CONTINUOUS
        self.env = WebAgentTextEnv(observation_mode="text", num_products=DEBUG_PROD_SIZE)

    def reset(self, next_subtask: str | None = None) -> Dict[str, str]:
        """Reset the environment and return the initial observation."""

        obs, _ = self.env.reset(next_subtask)
        available_actions = self.env.get_available_actions()
        return self._return_observation(obs, available_actions)

    def answer_parser(self, raw_response):
        return break_word_split("Action", raw_response).replace("`", "")

    def step(self, action: str) -> tuple[Dict[str, Any], float, bool]:
        """Perform an action and return the next observation, reward, and done."""

        obs, score, done, _ = self.env.step(action)
        available_actions = self.env.get_available_actions()
        return self._return_observation(obs, available_actions), score, done

    def _return_observation(self, obs: str, available_actions: Dict[str, Any]) -> Dict[str, str]:
        """Return the observation for the current step."""

        return {
            MemKey.OBSERVATION: obs,
            MemKey.AVAILABLE_ACTIONS: [
                "Has search bar: " + str(available_actions["has_search_bar"]),
                "\nClickables: " + str(available_actions["clickables"]),
            ],
        }

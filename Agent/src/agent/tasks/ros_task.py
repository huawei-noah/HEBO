from agent.tasks import Task
from .ros_api import RosApi
from typing import Any, Dict
import warnings


class ROSTask(Task):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ros_api = RosApi()
        print(self.ros_api)
        self.response = {"reward": 0.0,
                         "done": False,
                         "obs": ""}
        self.possible_actions = []

    def answer_parser(self, raw_response: str):
        return raw_response

    def is_complete(self):
        return self.response['done']

    def reset(self, next_subtask: str | None = None) -> Dict[str, str]:
        """Reset the environment and return the initial observation."""

        if next_subtask is not None:
            warnings.warn("ros_task does not support subtasks, ignoring subtask")

        response= self.ros_api.get_env_observation()

        return {
            "_text_obs": response,
            "_available_actions": self.possible_actions
        }

    def get_observation(self):
        obs = self.ros_api.get_env_observation()
        fdb = self.ros_api.get_feedback()
        return {
            "_available_actions": obs,
            "_available_actions": fdb
        }

    def step(self, action):
        print(action)
        self.ros_api.send_action(action)
        self.response = self.ros_api.receive_response()
        print(self.response['obs'])
        return {}, self.response["reward"], self.response["done"]


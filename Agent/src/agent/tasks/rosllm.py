import abc
from abc import ABC, abstractmethod
import traceback

import rospy
from rosllm_srvs.srv import Observation, ExecuteBehavior

from agent.memory import MemKey
from agent.tasks import Task


init_observation_str = """The environment observation for step t=0 is as follows.
{obs}
"""

action_output_str = """The output from the call to the "{prev_action_name}" function was:
{prev_action_output}

"""

observation_str = """The action chosen at t={prev_time_index} was as follows.
```json
{prev_action}
```
{action_output}
The environment observation for step t={time_index} is as follows.
{obs}
"""


class Extractor(ABC):
    @property
    @abstractmethod
    def pattern(self):
        """Pattern used to extract code."""

    @classmethod
    def extract(cls, response: str):
        return re.search(cls.pattern, response, re.DOTALL).group(1)


class JsonExtractor(Extractor):
    pattern = r"```json(.*?)```"


class ROSLLM(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rospy.init_node("agent_node")
        self.ros_resp = None
        self.step_index = None
        self.observations = []

    def answer_parser(self, raw):
        return raw

    def reset(self):
        self.step_index = 0
        return self.get_obs(init=True)

    def get_obs(self, init=False):
        rospy.wait_for_service("get_observation")
        try:
            get_obs = rospy.ServiceProxy("get_observation", Observation)
            obs = get_obs().observation
        except:
            err = traceback.format_exc()
            obs = f"[ERROR] failed to get observation from ROS, exception:\n{err}"

        if init:
            observation = init_observation_str.format(obs=obs)
        else:
            observation = observation_str.format(
                obs=obs,
                prev_action=self.action,
                prev_time_index=self.step - 1,
                time_index=self.step,
                action_output=self.get_action_output(),
            )

        self.observations.append(observation)

        return {MemKey.OBSERVATION: "\n".join(self.observations)}

    def get_action_output(self):
        if self.ros_resp.message:
            return action_output_str.format(prev_action_output=self.ros_resp.message)
        else:
            return ""

    def get_reward(self):
        failure = float(not self.ros_resp.success)
        return -1.0 * (1.0 + failure)

    def get_done(self):
        return rospy.is_shutdown()

    def execute_behavior(self):
        rospy.wait_for_service("execute_behavior")
        try:
            execute_behavior = rospy.ServiceProxy("execute_behavior", ExecuteBehavior)
            self.ros_resp = execute_behavior(self.action)
        except:
            pass

    def step(self, action):
        self.step += 1
        self.action = JsonExtractor.extract(action)
        self.execute_behavior()
        return self.get_obs(), self.get_reward(), self.get_done()

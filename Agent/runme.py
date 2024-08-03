import os

# from agent.tasks.rosllm import ROSLLM
from minillm.llm import LLM
from minillm import config_path
import rospy
import json
from abc import ABC, abstractmethod
import re
from rosllm_srvs.srv import Observation, ExecuteBehavior
import traceback
from copy import deepcopy
import time
import pickle

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


class ROSLLM:  # (Task):

    debug = True

    def __init__(self):
        # def __init__(self, **kwargs):
        # super().__init__(**kwargs)
        rospy.init_node("agent_node")
        self.ros_resp = None
        self.step_index = None
        self.observations = []
        self.task_descr = None
        self.data = []

    def answer_parser(self, raw):
        return raw

    def reset(self):
        # def reset(self, *args):
        self.step_index = 0
        print("Input task:")
        self.task_descr = input(">>")
        self.data.append((time.time_ns(), "USER_INPUT_TASK_DESCR"))
        return self.get_obs(init=True)

    def _parse_obs(self, observation):
        out = ""
        for obs in observation:
            for text in obs.text:
                out += f"* {text}\n"
        return out

    def get_obs(self, init=False):
        rospy.wait_for_service("get_observation")
        try:
            get_obs = rospy.ServiceProxy("get_observation", Observation)
            obs = self._parse_obs(get_obs().observation)
        except:
            err = traceback.format_exc()
            obs = f"[ERROR] failed to get observation from ROS, exception:\n{err}"

        if init:
            observation = init_observation_str.format(obs=obs)
        else:
            observation = observation_str.format(
                obs=obs,
                prev_action=self.action,
                prev_time_index=self.step_index - 1,
                time_index=self.step_index,
                action_output=self.get_action_output(),
            )

        self.observations.append(observation)

        return "\n".join(self.observations)  # {MemKey.OBSERVATION: "\n".join(self.observations)}

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

    def wait_for_approval(self):
        print(f"Step {self.step_index} action:")
        print("----")
        print(self.action)
        print("----")
        input("[TO CONTINUE PRESS ENTER]")

    def step(self, action):
        self.step_index += 1
        self.action = JsonExtractor.extract(action)
        if json.loads(self.action)["__action__"] == "done":
            print("--llm says we are done--")
            return self.get_obs(), self.get_reward(), True
        if self.debug:
            self.wait_for_approval()
        self.execute_behavior()
        return self.get_obs(), self.get_reward(), self.get_done()

    def save_data(self):
        stamp = time.time_ns()
        os.makedirs("data", exist_ok=True)
        path = f"data/rsl_demo_data_{stamp}.dat"
        with open(path, "wb") as f:
            pickle.dump(self.data, f)
        print("Saved", path)


prompt_preamble = """'{task_descr}'

The environment observations are listed below.

"""


def create_prompt(obs, task_descr):
    prompt_path = os.getcwd() + "/src/agent/prompts/templates/rsl/external_action.txt"
    with open(prompt_path, "r") as f:
        prompt = f.read()
    prompt += prompt_preamble.format(task_descr=task_descr)
    return prompt + obs


def load_llm():
    path = os.path.join(config_path, "deepseek.yaml")
    return LLM.load(path)


def main():
    env = ROSLLM()
    llm = load_llm()
    done = False
    obs = env.reset()
    while not done:
        prompt = create_prompt(obs, env.task_descr)
        env.data.append((time.time_ns(), "CREATED_PROMPT", deepcopy(prompt)))
        response = llm(prompt)
        env.data.append((time.time_ns(), "RECIEVED_RESPONSE", deepcopy(response)))
        print(f"======== Step {env.step_index+1} ========")
        print("--FULL PROMPT START--")
        print(prompt)
        print("--FULL PROMPT END--")
        print("--FULL RESPONSE START--")
        print(response)
        print("--FULL RESPONSE END--")
        obs, _, done = env.step(response)
        env.data.append((time.time_ns(), "FINISHED_STEP"))

    env.save_data()


if __name__ == "__main__":
    main()

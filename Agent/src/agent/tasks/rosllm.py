import traceback

import rospy
from rosllm_srvs.srv import Observation, ExecuteBehavior

from agent.memory import MemKey
from agent.tasks import Task


class ROSLLM(Task):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rospy.init_node("agent_node")
        self.ros_resp = None

    def reset(self):
        return self.get_obs()

    def get_obs(self):
        rospy.wait_for_service("get_observation")
        try:
            get_obs = rospy.ServiceProxy("get_observation", Observation)
            obs = get_obs().observation
        except:
            err = traceback.format_exc()
            obs = f"[ERROR] failed to get observation from ROS, exception:\n{err}"
        return {MemKey.OBSERVATION: obs}

    def get_reward(self):
        failure = float(not self.ros_resp.success)
        return -1.0 * (1.0 + failure)

    def get_done(self):
        return rospy.is_shutdown()

    def execute_behavior(self, action):
        rospy.wait_for_service("execute_behavior")
        try:
            execute_behavior = rospy.ServiceProxy("execute_behavior", ExecuteBehavior)
            self.ros_resp = execute_behavior(action)
        except:
            pass

    def step(self, action):
        self.execute_behavior(action)
        return self.get_obs(), self.get_reward(), self.get_done()

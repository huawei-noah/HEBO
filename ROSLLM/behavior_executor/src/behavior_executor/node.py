from typing import Tuple
from abc import ABC, abstractmethod

import rospy
from rosllm_srvs.srv import (
    ExecuteBehavior,
    ExecuteBehaviorResponse,
    ExecuteBehaviorRequest,
)
from behavior_executor.info import Info


class BehaviorExecutor(ABC):
    """Base class for a python implementation of a behavior executor."""

    def __init__(self) -> None:
        """Initialize node"""
        rospy.init_node(self.node_name)
        rospy.Service("execute_behavior", ExecuteBehavior, self.srv_callback)
        rospy.loginfo(f"{self.node_name} initialized")

    @property
    @abstractmethod
    def node_name(self) -> str:
        """Name for the node."""

    @staticmethod
    def init_request() -> Tuple[bool, str, Info]:
        success = True
        message = "successfully executed behavior"
        info = Info.OK
        return success, message, info

    @abstractmethod
    def execute_behavior(self, behavior: str) -> Tuple[bool, str, Info]:
        """Execute the behavior specified in the request in ROS."""

    def report(self, success: bool, message: str, info: Info) -> None:
        """Reports the success/failure of the behavior executor."""
        if success:
            rospy.loginfo(
                f"{self.node_name} processed request, behavior executed successfully"
            )
        else:
            msg = f"{self.node_name} failed to process request, behavior unsuccessful [{info=}]:\n{message}"
            if info.value > 0:
                rospy.logwarn(msg)
            else:
                rospy.logerr(msg)

    def srv_callback(self, req: ExecuteBehaviorRequest) -> ExecuteBehaviorResponse:
        """Main callback for the behavior executor service."""
        rospy.loginfo(f"{self.node_name} recieved request, executing behavior ...")
        success, message, info = self.execute_behavior(req.behavior)
        self.report(success, message, info)
        return ExecuteBehaviorResponse(
            success=success,
            message=message,
            info=info.value,
        )

    def spin(self):
        rospy.loginfo(f"spinning {self.node_name} ...")
        rospy.spin()


def main(node_cls: type):
    node_cls().spin()

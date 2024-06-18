import sys
from abc import ABC, abstractmethod

import rospy
from rosllm_srvs.srv import ExecuteBehavior, ExecuteBehaviorRequest


class TestBehaviorExecutor(ABC):

    srv_name = "execute_behavior"
    timeout = 30.0  # secs

    def __init__(self):
        rospy.init_node("test_behavior_executor")
        rospy.loginfo("initialized test_behavior_executor node")

    @property
    @abstractmethod
    def behavior(self) -> str:
        """The string that is used to define the behavior to be executed."""

    def call_service(self):
        """Calls the behavior executor service."""

        rospy.loginfo(f"calling service '{self.srv_name}'")

        # Wait for service
        rospy.wait_for_service(self.srv_name, timeout=self.timeout)

        # Execute behavior test
        try:
            handler = rospy.ServiceProxy(self.srv_name, ExecuteBehavior)
            req = ExecuteBehaviorRequest(behavior=self.behavior)
            resp = handler(req)
        except rospy.ServiceException as e:
            rospy.logwarn(f"call to service '{self.srv_name}' failed:\n{e}")
            ret = 1
            return ret

        # Report output
        if resp.success:
            ret = 0
            rospy.loginfo(
                f"executed behavior successfully:\n{resp.message}\n{resp.info=}"
            )
        else:
            ret = 1
            rospy.logerr(f"failed to execute behavior:\n{resp.message}\n{resp.info=}")

        # Complete
        rospy.loginfo("behavior execution test completed.")

        return ret


def main(node_cls: type):
    node = node_cls()
    return node.call_service()


if __name__ == "__main__":
    sys.exit(main())

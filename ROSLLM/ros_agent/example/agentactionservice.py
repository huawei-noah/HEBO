#!/usr/bin/env python
import rospy
from ros_agent.srv import HandleAgentAction, HandleAgentActionResponse

def handle_action(req):
    """
    Process the action request and return the response.
    This function simulates action processing by returning a success status,
    a message, and a reward.
    """
    print("Received action request: {}".format(req.action))
    
    # Here you would add the logic to process the action, e.g., controlling a robot or running an algorithm
    response_message = "Action processed successfully"
    reward = 1.0  # Example fixed reward; adjust based on actual action processing logic
    
    return HandleAgentActionResponse(success=True, response=response_message, reward=reward)

def action_service():
    rospy.init_node('agent_action_service')
    
    # Create the service 'handle_agent_action' and specify the handler function
    s = rospy.Service('handle_agent_action', HandleAgentAction, handle_action)
    
    print("Service 'handle_agent_action' ready to handle requests.")
    rospy.spin()  # Keep the service open.

if __name__ == "__main__":
    action_service()

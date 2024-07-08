#!/usr/bin/env python3
import rospy
import os
import signal

from ros_agent.srv import HandleAgentAction, HandleAgentActionRequest
from flask import Flask, request, jsonify

from std_srvs.srv import Trigger
from std_msgs.msg import String

class AgentApi:

    def __init__(self, node):
        self.node = node
        self.app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self):
        self.app.add_url_rule('/llmreq', 'handle_request_llm', self.handle_request_llm,  methods=['POST'])
        self.app.add_url_rule('/rosenv', 'handle_request_env', self.handle_request_env,  methods=['POST'])
        
    def handle_request_llm(self):
        if not request.json or 'action' not in request.json:
            return jsonify({"obs": "Request body must contain 'action' field",
                            "reward":0,'success':False}), 400

        action = request.json['action']
        success, resp, reward = self.node.handle_action(action)
        print("Agent response:",action)
        return jsonify({"done": success, "obs": resp, "reward": reward})

    def handle_request_env(self):
        print("handle env request OK")
        self.node.request_human_feedback()
        obs = self.node.get_combined_observation()
        # print(obs)
        return jsonify({"obs": obs, 'success': True if obs else False})
          

    def run_flask_app(self):
        self.app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)


class Node:


    def __init__(self):
        rospy.init_node("ros_agent_node", anonymous=True)
        self.srv_name = "handle_agent_action"
        self.latest_env_observation = "No observation yet"
        self.latest_human_feedback = "No human input yet"
        rospy.Subscriber("agent_environment", String, self.observation_callback)
        # rospy.Subscriber("human_feedback", String, self.feedback_callback)  # Subscribe to the human_feedback topic
        self.action_publisher = rospy.Publisher("agent_action", String, queue_size=10)
        self.api = AgentApi(self)
        self.api.run_flask_app()
        rospy.loginfo("initialized ros_agent node")

    def handle_action(self, action):
        try:
            self.action_publisher.publish(action)  # Publish the action to the ROS network
            handle_agent_action = rospy.ServiceProxy(self.srv_name, HandleAgentAction)
            req = HandleAgentActionRequest(action=action)
            resp = handle_agent_action(req)
            rospy.loginfo(f"Action response: {resp.response}")
            return resp.success, resp.response, resp.reward
        except rospy.ServiceException as e:
            success = False
            response = f"handling request failed: {e}"
            rospy.logwarn(f"handling request failed: {e}")
            return success, response, 0.0

    def request_human_feedback(self):
        """Prompt for human feedback from the terminal."""
        self.latest_human_feedback = input(" Human, please enter input: ")
        if self.latest_human_feedback.lower() == "exit":
            print("Killing program...")
            os.kill(os.getpid(), signal.SIGKILL)
        

    def observation_callback(self, msg):
        """Callback function to update the latest observed state."""
        self.latest_env_observation = msg.data

    # To read the humanfeedback from a topic 
    # def feedback_callback(self, msg):
    #     """Callback function to update the latest human feedback."""
    #     self.latest_human_feedback = msg.data

    
    def get_combined_observation(self):
        """Combine the latest observation with the latest human feedback."""
        combined = f"Environment Observation: {self.latest_env_observation} | Human Input: {self.latest_human_feedback}"
        print(f"Message to Agent: Environment Observation: {self.latest_env_observation} | Human Input: {self.latest_human_feedback} \n")
        return combined
    
    
    def spin(self):
        while not rospy.is_shutdown():
            self.request_human_feedback()
            rospy.spin()


def main():
    Node().spin()


if __name__ == "__main__":
    main()

#include <behaviortree_ros/bt_service_node.h>
#include <behaviortree_ros/bt_action_node.h>
#include <ros/ros.h>
#include <behaviortree_ros/AddTwoInts.h>
#include <behaviortree_ros/FibonacciAction.h>
#include <rosllm_srvs/ExecuteBehavior.h>
#include <iostream>

//
// Developer note
// ==============
//
// Currently, you can define ROS actions/services in Python or C++ in other packages.
// However, each action/service needs a wrapper implemented in this script so that
// the BehaviorTree.CPP library knows how to handle the execution of the action or
// service (i.e. did it fail or not).
//
// For now, I am leaving this as a requirement. I.e. we need to wrap each atomic action.
// However, there should be a better way to do this from a config file or something.
//
// One potential way of doing this is to specify a ROS action/service type that is then used
// for all atomic actions. Then it is the job of a user to implement their atomic action with one
// of these types.
//

using namespace BT;

//-------------------------------------------------------------
// Simple Action to print a number
//-------------------------------------------------------------

class PrintValue : public BT::SyncActionNode
{
public:
  PrintValue(const std::string& name, const BT::NodeConfiguration& config)
  : BT::SyncActionNode(name, config) {}

  BT::NodeStatus tick() override {
    int value = 0;
    if( getInput("message", value ) ){
      std::cout << "PrintValue: " << value << std::endl;
      return NodeStatus::SUCCESS;
    }
    else{
      std::cout << "PrintValue FAILED "<< std::endl;
      return NodeStatus::FAILURE;
    }
  }

  static BT::PortsList providedPorts() {
    return{ BT::InputPort<int>("message") };
  }
};

//-------------------------------------------------------------
// This client example is equal to this tutorial:
// http://wiki.ros.org/ROS/Tutorials/WritingServiceClient%28c%2B%2B%29
//-------------------------------------------------------------

class AddTwoIntsAction: public RosServiceNode<behaviortree_ros::AddTwoInts>
{

public:
  AddTwoIntsAction( ros::NodeHandle& handle, const std::string& node_name, const NodeConfiguration & conf):
  RosServiceNode<behaviortree_ros::AddTwoInts>(handle, node_name, conf) {}

  static PortsList providedPorts()
  {
    return  {
      InputPort<int>("first_int"),
      InputPort<int>("second_int"),
      OutputPort<int>("sum") };
  }

  void sendRequest(RequestType& request) override
  {
    getInput("first_int", request.a);
    getInput("second_int", request.b);
    expected_result_ = request.a + request.b;
    ROS_INFO("AddTwoInts: sending request");
  }

  NodeStatus onResponse(const ResponseType& rep) override
  {
    ROS_INFO("AddTwoInts: response received");
    if( rep.sum == expected_result_)
    {
      setOutput<int>("sum", rep.sum);
      return NodeStatus::SUCCESS;
    }
    else{
      ROS_ERROR("AddTwoInts replied something unexpected: %d", rep.sum);
      return NodeStatus::FAILURE;
    }
  }

  virtual NodeStatus onFailedRequest(RosServiceNode::FailureCause failure) override
  {
    ROS_ERROR("AddTwoInts request failed %d", static_cast<int>(failure));
    return NodeStatus::FAILURE;
  }

private:
  int expected_result_;
};

//-------------------------------------------------------------
// This client example is equal to this tutorial:
// http://wiki.ros.org/ROS/Tutorials/WritingServiceClient%28c%2B%2B%29
//-------------------------------------------------------------

class FibonacciServer: public RosActionNode<behaviortree_ros::FibonacciAction>
{

public:
  FibonacciServer( ros::NodeHandle& handle, const std::string& name, const NodeConfiguration & conf):
RosActionNode<behaviortree_ros::FibonacciAction>(handle, name, conf) {}

  static PortsList providedPorts()
  {
    return  {
      InputPort<int>("order"),
      OutputPort<int>("result") };
  }

  bool sendGoal(GoalType& goal) override
  {
    if( !getInput<int>("order", goal.order) )
    {
      // abourt the entire action. Result in a FAILURE
      return false;
    }
    expected_result_ = 0 + 1 + 1 + 2 + 3 + 5 + 8; // supposing order is 5
    ROS_INFO("FibonacciAction: sending request");
    return true;
  }

  NodeStatus onResult( const ResultType& res) override
  {
    ROS_INFO("FibonacciAction: result received");
    int fibonacci_result = 0;
    for( int n: res.sequence)
    {
      fibonacci_result += n;
    }
    if( fibonacci_result == expected_result_)
    {
      setOutput<int>("result", fibonacci_result);
      return NodeStatus::SUCCESS;
    }
    else{
      ROS_ERROR("FibonacciAction replied something unexpected: %d", fibonacci_result);
      return NodeStatus::FAILURE;
    }
  }

  virtual NodeStatus onFailedRequest(FailureCause failure) override
  {
    ROS_ERROR("FibonacciAction request failed %d", static_cast<int>(failure));
    return NodeStatus::FAILURE;
  }

  void halt() override
  {
    if( status() == NodeStatus::RUNNING )
    {
      ROS_WARN("FibonacciAction halted");
      BaseClass::halt();
    }
  }

private:
  int expected_result_;
};

class BehaviorTreeExecutorServer {
public:
    BehaviorTreeExecutorServer(ros::NodeHandle& nh) : nh_(nh) {
        service_ = nh_.advertiseService("execute_behavior", &BehaviorTreeExecutorServer::handleExecuteBehavior, this);
        ROS_INFO("execute_behavior server ready");
    }

private:
    bool handleExecuteBehavior(
        rosllm_srvs::ExecuteBehavior::Request &req,
        rosllm_srvs::ExecuteBehavior::Response &res
    ) {

        BehaviorTreeFactory factory;

        factory.registerNodeType<PrintValue>("PrintValue");
        RegisterRosService<AddTwoIntsAction>(factory, "AddTwoInts", nh_);
        RegisterRosAction<FibonacciServer>(factory, "Fibonacci", nh_);

        ROS_INFO("Behavior tree recieved:");
        std::cout << req.behavior << std::endl;
        auto tree = factory.createTreeFromText(req.behavior);

        NodeStatus status = NodeStatus::IDLE;

        while(status == NodeStatus::IDLE || status == NodeStatus::RUNNING)
        {
            status = tree.tickRoot();
            ROS_INFO("Stepped behavior tree, status:");
            std::cout << status << std::endl;
        }

        res.success = status == NodeStatus::SUCCESS;
        res.message = "finished executing behavior tree";
        res.info = 0;

        return true;
    }

    ros::NodeHandle& nh_;
    ros::ServiceServer service_;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "behavior_tree_executor_node");
  ros::NodeHandle nh;
  BehaviorTreeExecutorServer server(nh);
  ros::spin();
  return 0;
}

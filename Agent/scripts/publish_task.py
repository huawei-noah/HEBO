import csv
import hashlib
import json
import os
import pathlib
import random
from collections import defaultdict
from collections import deque
from enum import Enum
from pathlib import Path

import humanhash
import numpy as np
import pandas as pd
import redis
import typer
from rich import print
from typing_extensions import Annotated


def publish_task(redis_client, channel, task_data):
    # Convert the task data to a JSON string
    task_json = json.dumps(task_data)

    # Publish the task to the specified channel
    redis_client.publish(channel, task_json)


def one_hot_encode_agent(agent, values_dict):
    # Create a mapping of value to index in the one-hot vector
    value_to_index = {}
    index = 0
    for attr, values in values_dict.items():
        for value in values:
            value_to_index[(attr, value)] = index
            index += 1

    # Initialize a vector of zeros
    vector = [0] * index

    # Set the corresponding positions for the agent's attribute values to 1
    for attr, value in agent.items():
        if attr in values_dict:
            vector[value_to_index[(attr, value)]] = 1

    return vector


def resolve_features_for_agent(agent, agent_features_to_vec):
    resolved_features = {}

    for key, value in agent.items():
        # Check if the key is in the agent_features_to_vec dictionary
        if key in agent_features_to_vec:
            # Retrieve the corresponding vector
            feature_vector = agent_features_to_vec[key].get(value, None)

            # If the feature vector is found, add it to the resolved features
            if feature_vector is not None:
                resolved_features[key] = feature_vector

    return resolved_features


def hash_agent_features(features_dict):
    # Custom serializer for JSON
    def numpy_serializer(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        return obj  # Fallback to default behavior for other types

    # Convert the dictionary into a JSON string with sorted keys
    dict_string = json.dumps(features_dict, sort_keys=True, default=numpy_serializer)

    # Create a hash of the string
    hash_object = hashlib.sha256(dict_string.encode())
    return humanhash.humanize(hash_object.hexdigest())


def get_agent_group(agent, agents, distinct_agent_attributes):
    return "|".join([agents[agent]["agent"][k] for k in distinct_agent_attributes.keys()])


def append_to_csv(file_name, data, field_names):
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(file_name)

    with open(file_name, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


class UniformSamplingAgentSelector:
    def __init__(self):
        pass

    def select_agent_for_task(self, agent_features, task_features):
        """Selects an agent for a given task uniformly at random."""
        agent_ids = list(agent_features.keys())
        selected_agent_id = np.random.choice(agent_ids)
        return selected_agent_id

    def update_agent(self, agent_features, task_features, reward):
        """Updates the model with the performance of a specific agent for a given task.

        In uniform sampling, there's no update mechanism as selection is random.
        """
        pass


class HybridLinUcb:
    def __init__(self, alpha, shared_attr):
        self.alpha = alpha
        self.shared_attr = shared_attr

        # shared matrices:
        self.A0 = np.identity(3)
        self.b0 = np.zeros(3)

        self.As = {}  # np.identity(n_features)
        self.Bs = {}
        self.bs = {}  # np.zeros(n_features)

    def select_agent_for_task(self, agent_features, task_features):
        # for each agent we have non-shared features x and shared features z

        print(agent_features)

        pass

    def update_agent(self, agent_features, task_features, reward):
        pass


class LinUCBWithDisjointModels:
    def __init__(self, alpha):
        self.alpha = alpha

        # Initialize A as an identity matrix and b as a zero vector
        self.As = {}  # np.identity(n_features)
        self.bs = {}  # np.zeros(n_features)

    def select_agent_for_task(self, agent_features, task_features):
        """Selects an agent for a given task based on the task and agent features.

        agent_features: dict[str, dict[str,np.array[float]]]
        task_features: np.array[float]
        """

        # With disjoint linear models, we are only going to be looking at task features
        num_features = len(task_features)
        x = np.array(task_features)

        unique_agents = {
            hash_agent_features(features): (agent_id, features) for agent_id, features in agent_features.items()
        }
        unique_agent_features = {agent_id: features for _, (agent_id, features) in unique_agents.items()}

        p = {}
        for key, val in unique_agent_features.items():
            hashed_key = hash_agent_features(val)

            # Make sure As and bs are initialised for any new agents:
            if hashed_key not in self.As:
                self.As[hashed_key] = np.identity(num_features)
                self.bs[hashed_key] = np.zeros(num_features)

            A_inv = np.linalg.inv(self.As[hashed_key])
            theta = A_inv @ self.bs[hashed_key]
            p[key] = theta.T @ x + self.alpha * np.sqrt(x.T @ A_inv @ x)

        return max(p.keys(), key=lambda k: p[k])

    def update_agent(self, agent_features, task_features, reward):
        """Updates the model with the performance of a specific agent for a given task.

        agent_features: dict[str,np.array[float]]
        task_features: np.array[float]
        reward: float
        """

        x = np.array(task_features)
        hashed_key = hash_agent_features(agent_features)

        self.As[hashed_key] += np.outer(x, x)
        self.bs[hashed_key] += reward * x


class AgentSamplingChoices(str, Enum):
    disj_lin_ucb = "disj_lin_ucb"
    hybrid_lin_ucb = "hybrid_lin_ucb"
    uniform = "uniform"


def main(
        sampling_method: Annotated[AgentSamplingChoices, typer.Argument(show_default=False)],
        output_file: Annotated[Path, typer.Argument(show_default=False, dir_okay=False)],
        reward_threshold: float = 0.001,
        max_concurrent_tasks: int = 5,
        budget: float | None = None,
):
    random.seed(1337)
    list_of_costs = defaultdict(list)
    list_of_costs["gsm8k"] = [0.000123]
    list_of_costs["HumanEval"] = [0.000155]

    redis_client = redis.Redis(host="localhost", port=6379, db=0)

    agents = {
        key.decode("utf-8"): json.loads(value.decode("utf-8"))
        for key, value in redis_client.hgetall("agents_pool").items()
    }
    filepath = str(
        pathlib.Path(__file__).parent.resolve()) + "/resources/task_infos/all_task_infos_vecs_balanced_160_norm.json"
    with open(filepath) as f:
        task_to_vec = json.load(f)
    task_to_vec = dict(random.sample(list(task_to_vec.items()), 400))
    print(task_to_vec)

    print(f"Found {len(agents)} agents and {len(task_to_vec.keys())} tasks.")

    agent_attributes = list(a["agent"] for a in agents.values())
    distinct_agent_attributes = {
        k: list({d[k] for d in agent_attributes if k in d})
        for k in set().union(*agent_attributes)
        if len({d[k] for d in agent_attributes if k in d}) > 1 and k not in ("+seed", "task")
    }
    print(distinct_agent_attributes)

    agent_features_to_vec = {}
    for feature_key in distinct_agent_attributes.keys():
        values = distinct_agent_attributes[feature_key]
        agent_features_to_vec[feature_key] = {val: np.eye(len(values))[i] for i, val in enumerate(values)}

    if sampling_method == "disj_lin_ucb":
        agent_selector = LinUCBWithDisjointModels(alpha=0.5)
    elif sampling_method == "hybrid_lin_ucb":
        agent_selector = HybridLinUcb(alpha=0.4, shared_attr=["method"])
    elif sampling_method == "uniform":
        agent_selector = UniformSamplingAgentSelector()
    else:
        assert False
    print("Using agent selector: ", agent_selector)

    redis_client.delete("result_queue")

    task_queue = deque(task_to_vec.keys())
    available_agents: set[str] = set(agents.keys())
    active_tasks: int = 0
    task_reward: dict[str, float] = {}
    num_tokens = 0
    cost = 0
    task_attempts = 0
    logs = []

    while (task_queue or active_tasks) and (budget is None or budget > cost):
        # Assign tasks to available agents
        while task_queue and active_tasks < max_concurrent_tasks and available_agents:
            task_id = task_queue.popleft()
            if "alfworld" in task_id:
                continue
            valid_agents = {
                a for a in available_agents if agents[a]["task"].casefold() == task_id.split(":")[0].casefold()
            }
            assert len(valid_agents) > 0

            task_features = task_to_vec[task_id]
            agent_features = {
                vag_id: resolve_features_for_agent(agents[vag_id]["agent"], agent_features_to_vec)
                for vag_id in valid_agents
            }

            chosen_agent_id = agent_selector.select_agent_for_task(agent_features, task_features)

            active_tasks += 1
            available_agents.remove(chosen_agent_id)
            print(f"Pushing {task_id} to {chosen_agent_id}")
            redis_client.rpush(f"task_queue:{chosen_agent_id}", json.dumps({"subtask": task_id.split(":")[1]}))

        # process results
        _, results = redis_client.blpop("result_queue")
        ret_value = json.loads(results)

        agent_id = ret_value["agent"]
        env = agents[agent_id]["task"]
        task_id = ret_value["task"]

        cost = ret_value["result"]["cost"]
        avg_cost = cost if len(list_of_costs[env]) == 0 else (sum(list_of_costs[env]) / len(list_of_costs[env]))
        arm_reward = ret_value["result"]["discounted_returns"] / (cost / avg_cost)
        list_of_costs[env].append(cost)

        logs.append(
            {
                "agent_id": agent_id,
                "agent_group": get_agent_group(agent_id, agents, distinct_agent_attributes),
                "environment": env,
                "task_id": task_id,
                "discounted_returns": ret_value["result"]["discounted_returns"],
                "cost": ret_value["result"]["cost"],
                "num_input_tokens": ret_value["result"]["input_tokens"],
                "num_output_tokens": ret_value["result"]["output_tokens"],
                "arm_reward": arm_reward,
                "attempts": 1,
                "success": 1 if arm_reward > reward_threshold else 0,
            }
        )

        task_attempts += 1
        active_tasks -= 1

        available_agents.add(agent_id)

        agent_selector.update_agent(
            resolve_features_for_agent(agents[agent_id]["agent"], agent_features_to_vec),
            task_to_vec[agents[agent_id]["task"].lower() + ":" + task_id],
            arm_reward,
        )

        if arm_reward < reward_threshold:
            task_queue.append(agents[agent_id]["task"].lower() + ":" + task_id)
            print(f"Agent {agent_id[:6]} did not complete task {task_id}. Retrying...")
        else:
            print(f"Agent {agent_id[:6]} completed task {task_id} with reward {arm_reward:.3f}")
            task_reward[task_id] = arm_reward

        append_to_csv(
            output_file,
            logs[-1],
            field_names=[k for k in logs[-1].keys()],
        )
        df = pd.DataFrame.from_records(logs)

        for env in df["environment"].unique():
            print(env)
            print(
                df[df["environment"] == env]
                .drop(columns=["agent_id", "task_id", "environment"])
                .groupby("agent_group")
                .agg(
                    {
                        column: "sum" if column == "attempts" else "mean"
                        for column in df.columns
                        if column not in ["agent_id", "task_id", "environment", "agent_group"]
                    }
                )
            )
            print()

        cost = df["cost"].sum()

        print(f"Completed tasks: {len(task_reward)} out of {len(task_to_vec)}.")
        print(f"{task_attempts} total attempts and {num_tokens} total tokens (${cost:.2f}).")
    print(task_reward)


if __name__ == "__main__":
    typer.run(main)

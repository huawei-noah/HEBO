from agent.commands.core import Command
from agent.memory import MemKey


class SaveTrajectory(Command):
    name: str = "save_trajectory"
    description: str = "Save the full trajectory of the agent into memory."

    output_keys: dict[str, MemKey] = {
        "trajectory_mem_key": MemKey.TRAJECTORY,
        "raw_action_output_mem_key": MemKey.RAW_ACTION_OUTPUT,
    }

    def func(self, agent, trajectory_mem_key, raw_action_output_mem_key):
        if len(agent.llm.history) == 0:
            raise ValueError(
                "LLM history is expected to have at least one entry (for the action taken). "
                "Please make sure the history is being recorded if creating a new LLM backend."
            )

        new_traj_entry = []
        for step in agent.llm.history:
            full_agent_step = step["input"]
            answer = {"role": "assistant", "content": step["output"]}
            full_agent_step.append(answer)
            new_traj_entry.append(full_agent_step)

        # Store the raw llm output which generated the external action
        agent.memory.store(agent.llm.history[-1]["output"], {raw_action_output_mem_key})

        # Store this timestep's trajectory
        agent.memory.store(new_traj_entry, {trajectory_mem_key})
        agent.llm.history = []


class SaveEpisodeTrajectory(Command):
    name: str = "save_episode_trajectory"
    description: str = "Save the full current episode trajectory of the agent to memory."

    input_keys: dict[str, MemKey] = {"reward_mem_key": MemKey.REWARD, "trajectory_mem_key": MemKey.TRAJECTORY}
    output_keys: dict[str, MemKey] = {
        "full_eps_rewards_mem_key": MemKey.FULL_EPS_REWARDS,
        "full_eps_trajectory_mem_key": MemKey.FULL_EPS_TRAJECTORY,
    }

    def func(self, agent, reward_mem_key, trajectory_mem_key, full_eps_rewards_mem_key, full_eps_trajectory_mem_key):
        full_trajectory = agent.memory.retrieve_all({trajectory_mem_key: 1.0})[::-1]
        full_rewards = agent.memory.retrieve_all({reward_mem_key: 1.0})[::-1]

        assert len(full_trajectory) == len(full_rewards), "The length of the trajectory and rewards do not match."

        agent.memory.store(full_trajectory, {full_eps_trajectory_mem_key})
        agent.memory.store(full_rewards, {full_eps_rewards_mem_key})


class SendTrajectory(Command):
    name: str = "send_trajectory_to_server"
    description: str = "Extract trajectory from the memory and send it to the experience replay in the Redis server"

    input_keys: dict[str, MemKey] = {
        "full_eps_rewards_mem_key": MemKey.FULL_EPS_REWARDS,
        "full_eps_trajectory_mem_key": MemKey.FULL_EPS_TRAJECTORY,
    }
    output_keys: dict[str, MemKey] = {}

    def func(self, agent, full_eps_trajectory_mem_key, full_eps_rewards_mem_key):
        trajectory = agent.memory.retrieve({full_eps_trajectory_mem_key: 1.0})
        rewards = agent.memory.retrieve({full_eps_rewards_mem_key: 1.0})
        sample = {"trajectory": trajectory, "rewards": rewards}
        agent.llm.save_trajectory_to_redis(sample)

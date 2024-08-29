import multiprocessing

import hydra

from agent.train.utils import retrieve_training_tensors
from agent.utils.logger import FakeLogger


class EnvWorker(multiprocessing.Process):
    def __init__(self, cfg, llms, task_queue, data_queue, logger, worker_index, seed=0):
        super().__init__()
        self.cfg = cfg
        self.llms = llms
        self.task_queue = task_queue
        self.data_queue = data_queue
        self.logger = logger if worker_index == 0 else FakeLogger()
        self.seed = seed

        self.agents = [
            hydra.utils.instantiate(c, llm=llm, logger=self.logger) for c, llm in zip(self.cfg.agents, self.llms)
        ]
        self.task = hydra.utils.instantiate(
            self.cfg.task, seed=self.seed, agents=[self.agents[i] for i in self.cfg.task.agents]
        )
        for agent in self.agents:
            agent.agents = self.agents

    def collect_trajectories(self):
        total_steps = episode_count = 0
        queries, responses, masks, all_rewards = [], [], [], []
        while self.cfg.max_episodes is None or episode_count < self.cfg.max_episodes:
            episode_reward = 0
            self.task.reset()
            for agent in self.agents:
                agent.reset()

            timestep = 0
            while not self.task.is_complete() and (self.cfg.max_env_steps is None or timestep < self.cfg.max_env_steps):
                for agent in self.agents:
                    agent.observe(self.task)

                thinking_count = 0
                while self.cfg.max_thinking_steps is None or thinking_count < self.cfg.max_thinking_steps:
                    if all([agent.ready() for agent in self.task.agents]):
                        break
                    for agent in self.task.agents:
                        agent.step() if not agent.ready() else None

                    thinking_count += 1
                else:
                    for agent in self.task.agents:
                        agent.step() if not agent.ready() else None

                actions = {agent.name: agent.external_action for agent in self.task.agents}
                rew, done = self.task.step(actions)
                self.logger.log_metrics({"reward": rew})
                self.logger.save_metrics("all")
                timestep += 1
                episode_reward += rew

            used_assistant = False
            # assume single agent with simple trajectory for now
            for agent in self.agents:
                diag = agent.prompt_builder(["external_action.jinja"], {"memory": agent.memory})

                # check that the assistant is used at least once
                for msg in diag:
                    if msg["role"] == "assistant":
                        used_assistant = True
                        break

                if used_assistant:
                    query, resp, rmasks = retrieve_training_tensors(agent.llm.tokenizer, diag)

                    queries.append(query)
                    responses.append(resp)
                    masks.append(rmasks)

            if not used_assistant:
                continue

            total_steps += timestep
            all_rewards.append(episode_reward)
            episode_count += 1

        return total_steps, queries, responses, masks, all_rewards

    def run(self):
        super().run()

        while True:
            instruction = self.task_queue.get(block=True)
            if instruction == "stop":
                break

            total_steps, queries, responses, masks, all_rewards = self.collect_trajectories()

            self.data_queue.put((total_steps, queries, responses, masks, all_rewards))

            if isinstance(self.task_queue, multiprocessing.queues.JoinableQueue):
                self.task_queue.task_done()

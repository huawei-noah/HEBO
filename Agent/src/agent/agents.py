from typing import Any, Callable, Dict

from agent.commands import Flow
from agent.commands.flows import run_flow
from agent.memory import MemKey
from agent.memory import Memory
from agent.models.embeddings import EmbeddingBackend
from agent.models.llm import LanguageBackend
from agent.prompts.builder import PromptBuilder


class Agent:
    def __init__(self, name: str, version: str, logger):
        self.name = name
        self.version = version
        self.logger = logger

        self._external_action = None

    def reset(self) -> None:
        raise NotImplementedError

    def observe(self, obs) -> None:
        raise NotImplementedError

    def step(self) -> bool:
        raise NotImplementedError

    @property
    def external_action(self):
        return self._external_action

    @external_action.setter
    def external_action(self, value):
        self._external_action = value


class LLMAgent(Agent):
    def __init__(
        self,
        llm: Callable,
        embedding: Callable | None,
        memory: Memory,
        prompt_builder: PromptBuilder,
        on_init_flow: Flow | None,
        on_episode_start_flow: Flow | None,
        pre_action_flow: Flow,
        post_action_flow: Flow | None,
        on_episode_end_flow: Flow | None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.llm: LanguageBackend = llm(logger=self.logger)
        if embedding:
            self.embedding: EmbeddingBackend = embedding(logger=self.logger)

        self.memory = memory
        self.prompt_builder = prompt_builder
        self.prompt_builder.llm = self.llm

        self.on_init_flow = on_init_flow
        self.on_episode_start_flow = on_episode_start_flow
        self.pre_action_flow = pre_action_flow
        self.post_action_flow = post_action_flow
        self.on_episode_end_flow = on_episode_end_flow

        # TODO: can we improve this? e.g. set loggers upon instantiation
        self.llm.logger = self.logger
        self.memory.logger = self.logger
        self.prompt_builder.logger = self.logger

    def reset(self, task) -> None:
        """Reset the agent and its task."""

        self.memory.reset()
        self.task = task

        if hasattr(task, "task_category"):
            self.memory.store(task.task_category, {MemKey.TASK_CATEGORY})

    def observe(self, obs: Dict[str, Any]) -> None:
        """Store the observation in memory."""

        self.external_action = None
        for key, value in obs.items():
            self.memory.store(value, {key})

    def run_on_init_flow(self) -> None:
        run_flow(self.on_init_flow, self)

    def run_on_episode_start_flow(self) -> None:
        run_flow(self.on_episode_start_flow, self)

    def run_pre_action_flow(self) -> None:
        run_flow(self.pre_action_flow, self)
        assert self.external_action

    def run_post_action_flow(self) -> None:
        run_flow(self.post_action_flow, self)

    def run_on_episode_end_flow(self) -> None:
        run_flow(self.on_episode_end_flow, self)

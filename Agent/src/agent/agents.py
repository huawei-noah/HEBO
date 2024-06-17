from typing import Any, Callable, Dict, List, Union, Optional

import jellyfish

from agent.commands import Flow
from agent.commands.flows import run_flow
from agent.memory import MemKey
from agent.memory import Memory
from agent.models.embeddings import EmbeddingBackend
from agent.models.llm import LanguageBackend
from agent.parsers.parser import ParsingError
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
            self.memory.store(task.task_category, MemKey.TASK_CATEGORY)

    def observe(self, obs: Dict[str, Any]) -> None:
        """Store the observation in memory."""

        self.external_action = None
        for key, value in obs.items():
            self.memory.store(value, key)

    def run_on_init_flow(self) -> None:
        run_flow(self.on_init_flow, self)

    def run_on_episode_start_flow(self) -> None:
        run_flow(self.on_episode_start_flow, self)

    def run_pre_action_flow(self) -> None:
        run_flow(self.pre_action_flow, self)
        # assert self.external_action

    def run_post_action_flow(self) -> None:
        run_flow(self.post_action_flow, self)

    def run_on_episode_end_flow(self) -> None:
        run_flow(self.on_episode_end_flow, self)

    def safe_choose_from_options(self, ask_template: str, parse_func: Callable, format_error_message: str,
                                 options: List[str], prompt_kwargs: dict[str, Any] = None, max_retries: int = 5,
                                 human_takeover: bool = False, **kwargs) -> str | Dict[str, str]:

        """
        Same as safe_parsing_chat_completion but to choose from options.

        Args:
            human_takeover: whether human should provide the output if LLM keeps getting parsing errors
        """
        response = safe_parsing_chat_completion(
            agent=self, ask_template=ask_template, format_error_message=format_error_message,
            prompt_kwargs=prompt_kwargs,
            max_retries=max_retries, parse_func=parse_func, human_takeover=human_takeover, **kwargs
        )
        selected_option = min(options, key=lambda x: jellyfish.levenshtein_distance(response, x))
        self.llm.history[-1]["parsed_response"] = selected_option
        self.llm.logger.log_metrics({"llm:input_options": options, "llm:chosen_option": selected_option})
        return selected_option


class ThinkAndCodeLLMAgent(LLMAgent):
    """
    Agent with specifically two LLMs:
        one LLM for coding
        one LLM for reasoning
    """

    def __init__(
            self,
            llm: Callable,
            code_llm: Callable,
            embedding: Union[Callable, None],
            memory: Memory,
            prompt_builder: PromptBuilder,
            on_init_flow: Union[Flow, None],
            on_episode_start_flow: Union[Flow, None],
            pre_action_flow: Flow,
            post_action_flow: Union[Flow, None],
            on_episode_end_flow: Union[Flow, None],
            **kwargs,
    ):
        super().__init__(
            llm=llm,
            embedding=embedding,
            memory=memory,
            prompt_builder=prompt_builder,
            on_init_flow=on_init_flow,
            on_episode_start_flow=on_episode_start_flow,
            pre_action_flow=pre_action_flow,
            post_action_flow=post_action_flow,
            on_episode_end_flow=on_episode_end_flow,
            **kwargs,
        )
        # Add specialized LLMS to Agent
        self.specialized_llms: Dict[str, LanguageBackend] = {"code": code_llm(logger=self.logger)}


def safe_parsing_chat_completion(
        agent: LLMAgent,
        ask_template: str,
        parse_func: Callable,
        format_error_message: str,
        prompt_kwargs: dict[str, Any] = None,
        max_retries: int = 5,
        human_takeover: bool = False,
        specialized_llm_name: Optional[str] = None,
        **kwargs
) -> str | Dict[str, str]:
    """
    Tries agent.llm.chat_completion wrapped into an exception handling loop to safely catch ParsingErrors and retry
     a specified number of times.

    Args:
        human_takeover: whether human should provide the output if LLM keeps getting parsing errors
    """
    response = None
    _count = 0
    format_err_msg = None

    already_human_mode = agent.llm.human_mode

    while response is None and (_count < max_retries or human_takeover):
        prompt_builder_kwargs = {"memory": agent.memory, "format_err_msg": format_err_msg}
        if prompt_kwargs is not None:
            prompt_builder_kwargs.update(prompt_kwargs)

        llm_input = agent.prompt_builder(templates=[ask_template], kwargs=prompt_builder_kwargs)

        try:
            if _count >= max_retries:  # triggers human mode
                agent.llm.human_mode = True

            if specialized_llm_name is not None:
                assert isinstance(agent, ThinkAndCodeLLMAgent), \
                    (f"Trying to query a specialized llm but agent is not of type `ThinkAndCodeLLMAgent` or "
                     f"`ThinkAndCodeLLMAgent` but is of type `{type(agent)}`")
                assert hasattr(agent, 'specialized_llms'), \
                    f"Trying to query a specialized llm but agent doesn't have any."
                assert specialized_llm_name in agent.specialized_llms, \
                    f"Trying to query specialized llm {specialized_llm_name} but name not in `agent.specialized_llms`."
                response = agent.specialized_llms[specialized_llm_name].chat_completion(
                    messages=llm_input, parse_func=parse_func, **kwargs
                )
            else:
                response = agent.llm.chat_completion(messages=llm_input, parse_func=parse_func, **kwargs)

            if agent.llm.human_mode and not already_human_mode:  # if human_mode was triggered only for this loop
                agent.llm.human_mode = False

        except ParsingError as e:
            raw_response = e.args[1]
            if len(e.args) == 3:
                raw_error = e.args[2]
            else:
                raw_error = ""
            format_err_msg = (f'\nYour previous output was:'
                              f'\n-----\n{raw_response}\n-----\n'
                              f'which was not in the correct format.\n'
                              f'{raw_error}\n{format_error_message}')

        except Exception as e:
            raise e

        _count += 1

    if response is None:
        raise ParsingError(format_err_msg)

    return response

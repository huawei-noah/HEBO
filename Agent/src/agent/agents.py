from typing import Any, Callable, Union

import jellyfish
from jsonlines import jsonlines

from agent.commands import Flow
from agent.commands.flows import run_flow
from agent.memory import MemKey
from agent.memory import Memory
from agent.models.embeddings import EmbeddingBackend
from agent.models.llm import LanguageBackend
from agent.parsers.parser import ParsingError, UnsupportedExtensionError
from agent.prompts.builder import PromptBuilder


class Agent:
    def __init__(self, name: str, version: str, logger):
        self.name = name
        self.version = version
        self.logger = logger

        self._external_action = None
        self.task = None

    def reset(self) -> None:
        raise NotImplementedError

    def observe(self, obs) -> None:
        raise NotImplementedError

    def step(self) -> bool:
        raise NotImplementedError

    @property
    def external_action(self) -> Any:
        return self._external_action

    @external_action.setter
    def external_action(self, value) -> None:
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
            read_answer_from_file_path: str | list[str] | None = None,
            **kwargs,
    ):
        """
        Args:
            read_answer_from_file_path: whether to query the LLM or use the answers already saved a jsonl file,
                                        to activate in command-line: +agent.read_answer_from_file_path=...
        """
        super().__init__(**kwargs)

        self.llm_answers = self.get_llm_answers(read_answer_from_file_path=read_answer_from_file_path)
        self.llm: LanguageBackend = llm(logger=self.logger, responses_from_file=self.llm_answers)
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

    def chat_completion(self, llm: LanguageBackend, messages: list[dict[str, str]], parse_func: Callable,
                        **kwargs) -> str:

        """Generates a text completion for a given prompt in a chat-like interaction.

        Args:
            llm: the language backend to use
            messages: The input text prompt to generate completion for.
            parse_func: A function to parse the model's response.
            **kwargs: Additional keyword arguments that may be required for the generation,
                      such as temperature, max_tokens, etc.

        Returns:
            str: The generated text completion.
        """
        response = llm._chat_completion(messages=messages, parse_func=parse_func, **kwargs)
        self.set_human_mode(human_mode=llm.human_mode)
        return response

    @staticmethod
    def get_llm_answers(read_answer_from_file_path: str | list[str] | None) -> list[str] | None:
        if read_answer_from_file_path is None:
            return None
        elif isinstance(read_answer_from_file_path, str):
            return get_responses_from_file(log_path=read_answer_from_file_path)
        elif isinstance(read_answer_from_file_path, list):
            return read_answer_from_file_path

    def reset(self, task) -> None:
        """Reset the agent and its task."""

        self.memory.reset()
        self.task = task

        if hasattr(task, "task_category"):
            self.memory.store(task.task_category, MemKey.TASK_CATEGORY)

    def observe(self, obs: dict[MemKey, Any]) -> None:
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
                                 options: list[str], prompt_kwargs: dict[str, Any] = None, max_retries: int = 5,
                                 human_takeover: bool = False, **kwargs) -> str | dict[str, str]:

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

    def set_human_mode(self, human_mode: bool) -> None:
        """ Trigger human mode of all the LLMs """
        self.llm._human_mode = human_mode

    def is_in_human_mode(self) -> bool:
        """ Returns whether the llms of the agent are in human mode """
        return self.llm._human_mode


class MultiLLMAgent(LLMAgent):
    """
    Agent with a base LLM and a set of specific LLMs that can be called for specific tasks.
    For example, setting the argument `specialized_llms={'code': ...}`
        means that this specialized LLM is for code generation
    """

    def __init__(
            self,
            llm: Callable,
            specialized_llms: dict[str, Callable],
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
        self.specialized_llms: dict[str, LanguageBackend] = {
            name: llm(logger=self.logger, responses_from_file=self.llm_answers) for name, llm in
            specialized_llms.items()
        }

    def set_human_mode(self, human_mode: bool) -> None:
        """ Trigger human mode of all the LLMs """
        self.llm._human_mode = human_mode
        for k in self.specialized_llms:
            self.specialized_llms[k]._human_mode = human_mode

    def is_in_human_mode(self) -> bool:
        """ Returns whether the llms of the agent are in human mode """
        human_mode = self.llm._human_mode
        for k in self.specialized_llms:
            if human_mode != self.specialized_llms[k]._human_mode:
                raise RuntimeError(f"Human mode contradiction:"
                                   f"\n\t- main LLM is in human mode:{human_mode}"
                                   f"\n\t- {self.specialized_llms[k]} is in human_mode: {not human_mode}")
        return human_mode


class ThinkAndCodeLLMAgent(MultiLLMAgent):
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
            read_answer_from_file_path: str | list[str] | None = None,
            **kwargs,
    ):
        """
        Args:
            read_answer_from_file_path: to activate in command-line: +agent.read_answer_from_file_path=...
        """
        super().__init__(
            llm=llm,
            specialized_llms={"code": code_llm},
            embedding=embedding,
            memory=memory,
            prompt_builder=prompt_builder,
            on_init_flow=on_init_flow,
            on_episode_start_flow=on_episode_start_flow,
            pre_action_flow=pre_action_flow,
            post_action_flow=post_action_flow,
            on_episode_end_flow=on_episode_end_flow,
            read_answer_from_file_path=read_answer_from_file_path,
            **kwargs,
        )


def safe_parsing_chat_completion(
        agent: LLMAgent,
        ask_template: str,
        parse_func: Callable,
        format_error_message: str,
        prompt_kwargs: dict[str, Any] = None,
        max_retries: int = 5,
        human_takeover: bool = False,
        specialized_llm_name: str | None = None,
        **kwargs
) -> Union[str, dict[str, str]]:
    """
    Tries agent.chat_completion wrapped into an exception handling loop to safely catch ParsingErrors and retry
     a specified number of times.

    Args:
        human_takeover: whether human should provide the output if LLM keeps getting parsing errors
    """
    response = None
    _count = 0
    format_err_msg = None

    already_human_mode = agent.is_in_human_mode()

    while response is None and (_count < max_retries or human_takeover):
        prompt_builder_kwargs = {"memory": agent.memory, "format_err_msg": format_err_msg}
        if prompt_kwargs is not None:
            prompt_builder_kwargs.update(prompt_kwargs)

        llm_input = agent.prompt_builder(templates=[ask_template], kwargs=prompt_builder_kwargs)

        try:
            if _count >= max_retries:  # triggers human mode
                agent.set_human_mode(human_mode=True)

            if specialized_llm_name is not None:
                assert isinstance(agent, (MultiLLMAgent, ThinkAndCodeLLMAgent)), \
                    (f"Trying to query a specialized llm but agent is not of type `MultiLLMAgent` or "
                     f"`ThinkAndCodeLLMAgent` but is of type `{type(agent)}`")
                assert hasattr(agent, 'specialized_llms'), \
                    f"Trying to query a specialized llm but agent doesn't have any."
                assert specialized_llm_name in agent.specialized_llms, \
                    f"Trying to query specialized llm {specialized_llm_name} but name not in `agent.specialized_llms`."
                llm = agent.specialized_llms[specialized_llm_name]
                response = agent.chat_completion(llm=llm, messages=llm_input, parse_func=parse_func, **kwargs)
            else:
                llm = agent.llm
                response = agent.chat_completion(llm=llm, messages=llm_input, parse_func=parse_func, **kwargs)

            if agent.is_in_human_mode() and not already_human_mode:  # if human_mode was triggered only for this loop
                agent.set_human_mode(human_mode=False)

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

        except (FileNotFoundError, UnsupportedExtensionError) as e:
            err_msg = e.args[0]
            raw_response = e.args[1]
            format_err_msg = (f'\nYour previous output was:'
                              f'\n-----\n{raw_response}\n-----\n'
                              f'which contained erroneous paths.\n'
                              f'{err_msg}\n{format_error_message}')

        except Exception as e:
            raise e

        _count += 1

    if response is None:
        raise ParsingError(format_err_msg)

    return response


def get_responses_from_file(log_path: str) -> list[str]:
    """
    Get the list of LLM answers saved in a jsonl file.
    """
    history = []
    if "output.jsonl" not in log_path:
        log_path += "/output.jsonl"
    try:
        conv_history_path = log_path
        with jsonlines.open(conv_history_path) as f:
            conv = list(f)
            for i, row in enumerate(conv):
                if 'llm:output' in conv[i]:
                    if "@NOT GENERATED BY LLM AS IT WAS THE ONLY CHOICE@" in conv[i]['llm:output']:
                        continue
                    history.append(conv[i]['llm:output'])
    except FileNotFoundError:
        raise FileNotFoundError(f"No such file or directory when trying to run from logs: {log_path} ")
    return history

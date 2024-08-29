import pathlib
import re

from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import select_autoescape

from agent.memory import MemKey

_CONTEXT_LEN_BUFFER = 50
_SYS_AND_INITIAL_OBS_LEN = 2
_OBS_AND_ACT_PARITY = 2


class PromptBuilder:
    def __init__(self, template_paths: list[str], default_kwargs):
        self._base_template_path = pathlib.Path(__file__).parent.resolve() / "templates"
        self.template_paths = [self._base_template_path / tp for tp in template_paths]

        self.env = Environment(
            loader=FileSystemLoader(self.template_paths),
            autoescape=select_autoescape(),
        )

        self.llm = None
        self.max_trajectory_len = None
        self.default_kwargs = default_kwargs
        self.logger = None

    def __call__(self, templates, kwargs, add_msg=[]):
        kwargs = dict(**self.default_kwargs, **kwargs, max_trajectory_len=self.max_trajectory_len)

        while True:
            kwargs["max_trajectory_len"] = self.max_trajectory_len

            # generate the prompts
            rendered_prompts = []
            for template in templates:
                rendered_prompts.append(self.extract_messages(self.env.get_template(template).render(**kwargs)))

            messages = self.merge_messages(sum(rendered_prompts, []))
            messages.extend(add_msg)

            # Check if chat format is being using by looking to see if more than one message has the role of assistant
            is_chat_format = sum([1 for m in messages if m["role"] == "assistant"]) > 1

            # if the length is valid, break
            if self._check_length_valid(messages, is_chat_format) or "memory" not in kwargs:
                break

            # if the length is invalid, remove the oldest observation/action pair from trajectory
            if self.max_trajectory_len is None:
                # max trajectory length has not been set yet, so set it to the total trajectory length
                self.max_trajectory_len = len(kwargs["memory"].retrieve_all({MemKey.OBSERVATION: 1.0}))
            elif self.max_trajectory_len <= 0:
                # cannot reduce the trajectory length any further
                break

            self.max_trajectory_len -= 1
            print(f"Reducing max trajectory length to {self.max_trajectory_len}")

        if self.logger:
            self.logger.log_metrics({"templates": templates})

        return (
            (
                messages[:_SYS_AND_INITIAL_OBS_LEN]
                + messages[_SYS_AND_INITIAL_OBS_LEN:][-_OBS_AND_ACT_PARITY * (self.max_trajectory_len - 1) :]
            )
            if is_chat_format and self.max_trajectory_len
            else messages
        )

    def _check_length_valid(self, messages, is_chat_format):
        """Check if the length of the messages is less than the maximum prompt length."""
        return True
        # max_prompt_len = self.llm.context_length - 300  # - self.llm.max_response_tokens
        # return self.llm.count_tokens(messages) < max_prompt_len - _CONTEXT_LEN_BUFFER

    def extract_messages(self, prompt):
        split_prompt = re.findall(r"\[\[ (\w+) \]\]\n((?:.|\n)*?(?=(?:\n\[\[|$)))", prompt)
        messages = []
        for role, content in split_prompt:
            messages.append({"role": role.lower(), "content": content})
        return messages

    def merge_messages(self, messages):
        cur_role = None
        cur_content = ""
        new_messages = []
        for msg in messages:
            if msg["role"] == cur_role:
                cur_content += msg["content"]
            else:
                if cur_role:
                    new_messages.append({"role": cur_role, "content": cur_content})
                cur_content = msg["content"]
                cur_role = msg["role"]
        new_messages.append({"role": cur_role, "content": cur_content})
        return new_messages

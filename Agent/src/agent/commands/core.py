from typing import Annotated, Any

from pydantic import BaseModel
from pydantic import ConfigDict

from agent.memory import MemKey


class Command(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    name: Annotated[
        str,
        "The name of the command. Most commonly used when asking a "
        "language model what command to execute next (e.g., in flows).",
    ]
    description: Annotated[
        str,
        "The description of the command. Most commonly used when asking a "
        "language model what command to execute next (e.g., in flows).",
    ]

    # inputs/output keys and prompts are keyword argument inputs to func()

    input_keys: Annotated[
        dict[str, MemKey],
        (
            "Memory keys that are expected to be in the memory when the function is executed.\n\n"
            "The _key_ to the dictionary is the argument name and the _value_ is passed as an argument value.\n"
            'For example, a dictionary such as {"key": "value"} means that func will be called as func(key=value).\n\n'
            "The keyword arguments are collected from all of input_keys, output_keys, and required_prompt_templates.\n"
            "For this reason, keys have to be unique among input_keys, output_keys and required_prompt_templates."
        ),
    ] = {}

    output_keys: Annotated[
        dict[str, MemKey],
        (
            "Memory keys that are going to be created or altered when the function is executed."
            "The _key_ to the dictionary is the argument name and the _value_ is passed as an argument value.\n"
            "See documentation of input_keys for an explanation of how this is used."
        ),
    ] = {}

    required_prompt_templates: Annotated[
        dict[str, str],
        (
            "Prompt templates used by this function. These templates need to exist within the searchable paths",
            "The _key_ to the dictionary is the argument name and the _value_ is passed as an argument value.\n",
            "See documentation of input_keys for an explanation of how this is used.",
        ),
    ] = {}

    # operations that will happen on the memory (e.g. renames or deletions).
    # this is for convenience and may be done manually if reported as input/output keys.
    renamed_keys: Annotated[
        dict[MemKey, MemKey],
        "Memory keys that will be renamed. The key of this dictionary is the source name and the value is the new name."
        "Renaming happens after the main function is executed, but before deletions",
    ] = {}

    deleted_keys: Annotated[
        list[MemKey], "Memory keys that will be deleted. Deletion happens at the end of the function call."
    ] = []

    # optional documentation of memory keys and templates
    # only used for building docs
    docs: Annotated[
        dict[str, str],
        (
            "Optional documentation of input_keys, output_keys, and required_prompt_templates.\n"
            "The keys between this dictionary and the ones being documented must match."
        ),
    ] = {}

    def __call__(self, agent):
        """This function is the entrypoint of this (in/ex)trinsic function.

        The keys below all related to the agent's memory:
            input_keys: need to be in the memory when the command is called.
            renamed_keys: will be renamed right after func() is called.
            deleted_keys: will be deleted right after the other keys are renamed.
            output_keys: need to be in the memory at the end of this function.
        """

        # This ensures that no key is shared across any of the inputs
        # If it is shared, it results to undefined behaviour as we cannot guarantee
        # which of the keys are going to pass to the func
        assert not set(self.input_keys) & set(self.output_keys)
        assert not set(self.input_keys) & set(self.required_prompt_templates)
        assert not set(self.output_keys) & set(self.required_prompt_templates)

        # builds the inputs to the func()
        func_kwargs = {**self.input_keys, **self.output_keys, **self.required_prompt_templates}

        self.func(agent, **func_kwargs)

        for source, target in self.renamed_keys.items():
            agent.memory.rename(source, target)

        agent.memory.delete(self.deleted_keys)

    # Commands need to define this method:
    def func(self, agent, *args: Any, **kwargs: Any):
        ...


class DoNothing(Command):
    name: str = "do_nothing"
    description: str = "Do nothing, skip a turn, NoOp."

    def func(self, agent):
        """This function doesn't do anything."""
        pass


class UseTool(Command):
    name: str = "use_tool"
    description: str = "Generate some inputs to a specific tool in a correct format"

    tool: Any

import pathlib

from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import select_autoescape

from agent.prompts import builder as prompt_builder_lib


def test_role_content_extraction_1():
    _base_template_path = pathlib.Path(__file__).parent.resolve()
    env = Environment(
        loader=FileSystemLoader(_base_template_path),
        autoescape=select_autoescape(),
    )
    builder = prompt_builder_lib.PromptBuilder([], {})
    prompt = env.get_template("webarena_test_template.jinja").render()
    messages = builder.extract_messages(prompt)

    print(prompt)
    print(messages)
    assert len(messages) == 5
    assert messages[0]["content"].startswith("You are an")
    assert messages[0]["role"] == "system"

from agent.commands.core import Command
from agent.memory import MemKey


class ChunkSummarization(Command):
    """Use text retrieved to summarise it in smaller text chunk by chunk."""

    name: str = "summarise_text"
    description: str = "Summarise text chunks to reduce their size"

    required_prompt_templates: dict[str, str] = {"ask_template": "summarize_chunks.jinja"}
    input_keys: dict[str, str] = {"text_retrieved_key": MemKey.TEXT_RETRIEVED, "num_of_chunks_key": MemKey.NUM_CHUNKS}
    output_keys: dict[str, str] = {"summarized_text_key": MemKey.SUMMARIZED_TEXT}

    def func(self, agent, ask_template, text_retrieved_key, num_of_chunks_key, summarized_text_key):
        num_chunks = int(agent.memory.retrieve({num_of_chunks_key: 1.0}))
        text_chunks = agent.memory.retrieve_all({text_retrieved_key: 1.0})[:num_chunks]
        for text in text_chunks:
            summarization_prompt = agent.prompt_builder([ask_template], {"text": text})
            summarized_text = agent.llm.chat_completion(summarization_prompt, lambda x: x)
            agent.memory.store(summarized_text, {summarized_text_key})

import json
import logging
from functools import partial

import accelerate
import hydra
import mlrq
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

from agent.train.replay_buffer import ReplayBuffer
from agent.train.replay_buffer import add_trajectory
from agent.train.replay_buffer import postprocess

logger = logging.getLogger(__name__)

accelerator = accelerate.Accelerator()


@mlrq.distribute(
    max_batch_size=4,
    batch_on=["messages"],
)
def chat_completion(
    model,
    tokenizer,
    messages,
    generation_kwargs,
):
    logger.info(f"Received {len(messages)} chat_completion jobs. Processing...")
    b_prompt = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

    inputs = tokenizer(b_prompt, return_tensors="pt", padding="longest").to(accelerator.device)

    gen_outputs = model.generate(
        **inputs,
        **generation_kwargs,
        return_dict_in_generate=True,
        use_cache=True,
    )
    generations = gen_outputs["sequences"]
    generations = generations[:, inputs["input_ids"].shape[1] :].to("cpu")  # remove prompt

    decoded = tokenizer.batch_decode(generations, skip_special_tokens=True)

    logger.info(f"Finished processing {len(decoded)} chat_completion jobs.")
    return decoded


@mlrq.distribute()
def count_tokens(
    tokenizer,
    messages,
):
    prompt = tokenizer.apply_chat_template(json.loads(messages), tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.encode(prompt)

    logger.info(f"Counting tokens... {len(tokens)} found.")
    return len(tokens)


@hydra.main(version_base="1.3", config_path="../../../configs/training/", config_name="server_training")
def main(cfg: DictConfig) -> None:
    rclient = hydra.utils.instantiate(cfg.client)

    model_id = cfg.model.model_id
    model = AutoModelForCausalLM.from_pretrained(model_id, **cfg.model.model_kwargs, device_map=accelerator.device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right", **cfg.model.tokenizer_kwargs)

    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model.resize_token_embeddings(len(tokenizer))

    replay_buffer = ReplayBuffer(cfg.training.on_policy, cfg.training.dataset_size, cfg.training.update_frequency)
    response_template = cfg.model.response_template
    instruction_template = cfg.model.instruction_template
    response_template_tokenized = tokenizer.encode(f"{response_template}", add_special_tokens=False)
    instruction_template_tokenized = tokenizer.encode(f"{instruction_template}", add_special_tokens=False)
    mask_generator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template_tokenized,
        response_template=response_template_tokenized,
        tokenizer=tokenizer,
    )

    collator = partial(postprocess, tokenizer=tokenizer, format="chat", mask_gen=mask_generator)
    dataloader = DataLoader(replay_buffer, batch_size=cfg.training.batch_size_per_worker, collate_fn=collator)
    worker = mlrq.Worker(
        rclient,
        implements={
            chat_completion: {"model": model, "tokenizer": tokenizer},
            count_tokens: {"tokenizer": tokenizer},
            add_trajectory: {"replay_buffer": replay_buffer},
        },
    )
    while True:
        worker.run(
            stop_condition=lambda: len(replay_buffer) % cfg.training.update_frequency == 0 and len(replay_buffer) > 0
        )
        for batch in dataloader:
            print(batch)


if __name__ == "__main__":
    main()

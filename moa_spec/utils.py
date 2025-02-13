# MIT License
#
# Copyright (c) 2024, Huawei Technologies Co., Ltd
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import os
import random
from dataclasses import dataclass

import hydra
import numpy as np
import torch
from datasets import Dataset
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from trl import DataCollatorForCompletionOnlyLM

logger = logging.getLogger(__name__)


def setup_distributed_environment(seed: int) -> tuple[int, int, int, str]:
    seed_gpu_wise = seed
    local_rank = 0
    world_size = 1

    if os.environ.get("LOCAL_RANK") is not None:
        # distributed setup, accelerator object will be built inside the Trainer
        local_rank = int(os.environ.get("LOCAL_RANK"))
        world_size = int(os.environ.get("WORLD_SIZE"))
        seed_gpu_wise += local_rank

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_map = f"cuda:{local_rank}"
    else:
        torch.npu.set_device(local_rank)
        device_map = f"npu:{local_rank}"

    return local_rank, world_size, seed_gpu_wise, device_map


def set_and_print_config(cfg, rank=0):
    OmegaConf.resolve(cfg)

    if rank == 0:
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        logger.info(f"Output dir: {output_dir}")
        logger.info(f"Configuration:\n"
                    f"{OmegaConf.to_yaml(cfg)}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_dataset(dataset, tokenizer, total_number_tokens, max_length=2048, rank=0):
    input_texts = []
    labels = []
    ignore_index = -100
    all_length = []
    no_response_token = 0
    shorten_sequence = 0
    formatted_examples = []

    # extract pattern
    unique_token = tokenizer.convert_ids_to_tokens(total_number_tokens - 1)
    chat_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": unique_token}, {"role": "assistant", "content": unique_token}], tokenize=False
    )
    instruction_template, response_template = chat_str.split(unique_token)[:2]
    response_template_tokens = tokenizer(response_template, add_special_tokens=False).input_ids

    data_collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        ignore_index=ignore_index,
    )

    if rank == 0:
        logger.info("Apply chat template...")

    # Prepare data with instruct format
    for item in tqdm(dataset) if rank == 0 else dataset:
        if "messages" in item.keys():
            msg = item["messages"]
        elif "history" in item.keys() and item["history"] is not None:
            msg = item["history"]
        elif "conversations" in item.keys():
            msg = item["conversations"]
            msg = [{"content": m["value"], "role": "assistant" if m["from"] == "gpt" else "user"} for m in msg]
        else:
            msg = [
                {"content": item["instruction"].strip(), "role": "user"},
                {"content": item["output"].strip(), "role": "assistant"},
            ]

        formatted_example = tokenizer.apply_chat_template(msg, tokenize=False)
        formatted_examples.append(formatted_example)

    if rank == 0:
        logger.info("Tokenize...")

    all_input_ids = tokenizer(formatted_examples, truncation=False, padding=False, add_special_tokens=False).input_ids

    if rank == 0:
        logger.info("Filter by size and set label...")

    for input_ids in tqdm(all_input_ids) if rank == 0 else all_input_ids:

        # gather stats for length
        all_length.append(len(input_ids))
        assert total_number_tokens - 1 not in input_ids

        if len(input_ids) > max_length:
            shorten_sequence += 1

        input_ids = input_ids[:max_length]

        if not is_consecutive_subsequence(input_ids, response_template_tokens):
            no_response_token += 1
            continue

        out = data_collator([input_ids])

        if not (ignore_index != out['labels']).any():
            no_response_token += 1
            continue

        input_texts.append(out['input_ids'][0].tolist())
        labels.append(out['labels'][0].tolist())

    if rank == 0:
        logger.info("Convert to torch tensors...")

    dataset = Dataset.from_dict({'input_ids': input_texts, 'labels': labels}).with_format("pytorch")

    if rank == 0:
        logger.info(f"Used max length: {max_length}")
        logger.info(f"Average length: {np.mean(all_length)}, "
                    f"remove {no_response_token} samples, "
                    f"and {shorten_sequence} shorten (including the removed).")
        logger.info(f"Remaining datapoint: {len(input_texts)}")

    assert len(input_texts) > 0
    return dataset


def is_consecutive_subsequence(list_a, list_b):
    """
    Check if list_b is a consecutive subsequence of list_a
    """
    # Get the lengths of both lists
    len_a = len(list_a)
    len_b = len(list_b)

    # If list_b is longer than list_a, it can't be a subsequence
    if len_b > len_a:
        return False

    # Slide over list_a with a window of size len_b
    for i in range(len_a - len_b + 1):
        # Extract the subsequence from list_a
        subsequence = list_a[i:i + len_b]
        # Check if this subsequence matches list_b
        if subsequence == list_b:
            return True

    return False


@dataclass
class DataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    ignore_index = -100

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [torch.ones_like(item['input_ids'], dtype=torch.int32) for item in batch]
        response_mask = [(item['labels'] != -100).int() for item in batch]

        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        padded_response_mask = pad_sequence(response_mask, batch_first=True, padding_value=0)

        return padded_input_ids, padded_attention_mask, padded_response_mask


def convert_list_of_dicts_to_dict_of_lists(all_stats):
    # Initialize an empty dictionary to hold lists for each key
    dict_of_lists = {}

    # Iterate over each dictionary in the list
    for stats in all_stats:
        for key, value in stats.items():
            # If the key is not already in the dictionary, initialize it with an empty list
            if key not in dict_of_lists:
                dict_of_lists[key] = []
            # Append the value to the corresponding list
            dict_of_lists[key].append(value)

    return dict_of_lists


def zero_peak_constant_scheduler(zero_warmup_steps, decay_steps, peak_lr, constant_lr):
    assert zero_warmup_steps <= decay_steps, f"{zero_warmup_steps} should be lower or equals than {decay_steps}"

    def lr_lambda(current_step):
        if current_step < zero_warmup_steps:
            # Phase 1: Linear warmup from 0 to peak_lr
            return (peak_lr / constant_lr) * (current_step / zero_warmup_steps)
        elif current_step < zero_warmup_steps + decay_steps:
            # Phase 2: Linear decay from peak_lr to constant_lr
            decay_progress = (current_step - zero_warmup_steps) / decay_steps
            return (peak_lr / constant_lr) - ((peak_lr - constant_lr) / constant_lr) * decay_progress
        else:
            # Phase 3: Constant at constant_lr
            return 1.0
    return lr_lambda

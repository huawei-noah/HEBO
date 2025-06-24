# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, List, Literal, Optional

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError

from .configs import DataArguments
import numpy as np
import random


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def maybe_insert_system_message(messages, tokenizer):
    if messages[0]["role"] == "system":
        return

    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.default_chat_template

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template or "<|im_start|>" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo", "tdpo", "simpo", "mapo", "sparse", "ad-sparse", "smaug"],
    auto_insert_empty_system_msg: bool = True,
):
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(chosen_messages, tokenizer)
                maybe_insert_system_message(rejected_messages, tokenizer)

            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task in ["dpo", "tdpo", "orpo", "sparse", "simpo", "mapo", "ad-sparse", "smaug"]:
        if all(k in example.keys() for k in ("chosen", "rejected")):
            if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
                raise ValueError(
                    f"Could not format example as dialogue for `{task}` task! Require OpenAI format for all messages"
                )

            # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            if "prompt" in example and is_openai_format(example["prompt"]):
                prompt_messages = example["prompt"]
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
            else:
                prompt_messages = example["chosen"][:-1]
                # Now we extract the final turn to define chosen/rejected responses
                chosen_messages = example["chosen"][-1:]
                rejected_messages = example["rejected"][-1:]

            # Prepend a system message if the first message is not a system message
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)

            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `{task}` task! Require either the "
                f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of ['sft', 'generation', 'rm', 'dpo', 'orpo']"
        )
    return example


def is_openai_format(messages: Any) -> bool:
    """
    Check if the input messages are in OpenAI format.
    Args:
        messages (`Any`):
            Messages to check.
    Returns:
        `bool`: Whether the messages are in OpenAI format.
    """
    if isinstance(messages, list) and all(isinstance(message, dict) for message in messages):
        return all("role" in message and "content" in message for message in messages)
    return False


def get_datasets(
    data_config: DataArguments | dict,
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle: bool = True,
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.

    Args:
        data_config (`DataArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'data_config' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.

    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """
    if type(data_config) is DataArguments:
        # Structure of the config to read the datasets and their mix
        # datasets_mixer:
        #     - 'dataset1': 0.5
        #     - 'dataset2': 0.3
        #     - 'dataset3': 0.2
        dataset_mixer = data_config.dataset_mixer
    elif isinstance(data_config, dict):
        # Structure of the input is:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset1": 0.3,
        #             "dataset1": 0.2,
        #         }
        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config {data_config} not recognized.")

    raw_datasets = mix_datasets(
        dataset_mixer,
        splits=splits,
        configs=configs,
        columns_to_keep=columns_to_keep,
        shuffle=shuffle,
    )
    return raw_datasets


def get_imdb_odpo(filename_or_hub_name, split):
    def split_prompt_and_responses(sample):
        prompt = sample["prompt"]
        return {
            # "messages": [{"content": sample["chosen"]["text"][len(sample['text']):], "role": "user"} ],
            "prompt": prompt,
            "chosen_rw": sample["chosen"]["score"],
            "rejected_rw": sample["rejected"]["score"],
            "chosen": [
                {"content": prompt, "role": "user"},
                {"content": sample["chosen"]["text"], "role": "assistant"},
            ],
            "rejected": [
                {"content": prompt, "role": "user"},
                {"content": sample["rejected"]["text"], "role": "assistant"},
            ],
        }
    #
    dataset = load_dataset("json", data_files=os.path.join(filename_or_hub_name, f"imdb-po-{split}.json"), split='train')
    return dataset.map(split_prompt_and_responses)


def get_tldr_odpo(filename_or_hub_name, split):
    def split_prompt_and_responses(sample):
        prompt = sample["prompt"]
        return {
            # "messages": [{"content": sample["chosen"]["text"][len(sample['text']):], "role": "user"} ],
            "prompt": prompt,
            "chosen": [
                {"content": prompt, "role": "user"},
                {"content": sample["chosen"]["text"], "role": "assistant"},
            ],
            "rejected": [
                {"content": prompt, "role": "user"},
                {"content": sample["rejected"]["text"], "role": "assistant"},
            ],
        }
    dataset = load_dataset("json", data_files=os.path.join(filename_or_hub_name, f"tldr-po-{split}.json"), split='train')
    return dataset.map(split_prompt_and_responses)


def get_tldr_combined(filename_or_hub_name, split, config=None):
    def split_prompt_and_responses(sample):
        prompt = sample["prompt"]
        return {
            # "messages": [{"content": sample["chosen"]["text"][len(sample['text']):], "role": "user"} ],
            "prompt": prompt,
            "chosen": [
                {"content": prompt, "role": "user"},
                {"content": sample["chosen"], "role": "assistant"},
            ],
            "rejected": [
                {"content": prompt, "role": "user"},
                {"content": sample["rejected"], "role": "assistant"},
            ],
        }
    config_pref = f"-{config}" if config is not None else ""
    dataset = load_dataset("json", data_files=os.path.join(filename_or_hub_name,f"tldr-po-{split}{config_pref}.json"), split='train')
    return dataset.map(split_prompt_and_responses)



def mix_datasets(
    dataset_mixer: dict,
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle=True,
) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'dataset_mixer' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    """
    splits = ["train", "test"] if splits is None else splits
    configs = [None] * len(dataset_mixer) if not configs else configs
    columns_to_keep = [] if columns_to_keep is None else columns_to_keep

    if configs is not None and len(configs) != len(dataset_mixer):
        raise ValueError("The number of given dataset config names must be the same as the given number of datasets.")

    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for (ds, frac), ds_config in zip(dataset_mixer.items(), configs):
        fracs.append(frac)

        for split in splits:

            # imdb odpo
            if "imdb" in ds:
                dataset = get_imdb_odpo(ds, split)
            
            elif "tldr" in ds:
                dataset = get_tldr_combined(ds, split, ds_config)

            # argyla 7k
            elif 'dpo-mix-7k' in ds:
                #
                def transform(samples):
                    new = {"chosen": [], "rejected": [], "prompt": []}

                    for win, lose in zip(samples['chosen'], samples['rejected']):
                        if win[0]['role'] == 'user':
                            prompt = win[0]['content']
                        else:
                            prompt = win[1]['content']

                        new["chosen"].append(win)
                        new["rejected"].append(lose)
                        new["prompt"].append(prompt)
                    return new

                dataset = load_dataset("parquet", data_files=os.path.join(ds, "data", f"{split}-*.parquet"), split='train')
                dataset = dataset.map(transform, batched=True, remove_columns=dataset.column_names)

            # anthropic HH
            elif "hh" in ds:
                #
                def split_roles(text):
                    chunks = [x for x in text.split("\n\n") if x!='']
                    res = []
                    for cc in chunks:
                        if cc.startswith("Human: "):
                            res.append({
                                'content': cc[7:],
                                'role': 'user'
                            })
                        elif cc.startswith("Assistant: "):
                            res.append({
                                'content': cc[11:],
                                'role': 'assistant'
                            })
                        else:
                            res[-1]['content'] = '\n\n'.join([res[-1]['content'],cc])
                    return res
                #
                def transform(samples):
                    new = {"chosen": [], "rejected": [], "prompt": [], "messages":[]}

                    for win, lose in zip(samples['chosen'], samples['rejected']):
                        chosen = split_roles(win)
                        if len(chosen)==0: continue

                        new["chosen"].append(chosen)
                        new["rejected"].append(split_roles(lose))
                        new["prompt"].append(chosen[0]['content'])
                        new["messages"].append(chosen)
                    return new
                #
                dataset = load_dataset(ds, split=split)
                dataset = dataset.map(transform, batched=True, remove_columns=dataset.column_names)

            # Milan's code dataset
            elif "mbpp_new" in ds:
                #
                def transform(samples):
                    new = {"chosen": [], "rejected": [], "prompt": []}

                    for ex, prompt in zip(samples['answers'], samples['question']):
                        responses, scores = [], []
                        for r in ex:
                            responses.append(r['text'])
                            scores.append(np.inf if r['votes'] is None else float(r['votes']))

                        responses, scores = np.array(responses), np.array(scores)
                        mask = np.isfinite(scores)

                        if split != 'train':
                            # select only 1 for validation set
                            chosen_response = responses[random.choice(np.flatnonzero(mask))]
                            rejected_response = responses[random.choice(np.flatnonzero(~mask))]

                            new["chosen"].append(chosen_response)
                            new["rejected"].append(rejected_response)
                            new["prompt"].append(prompt)
                        else:
                            # rollout all data
                            for chosen_response in responses[np.flatnonzero(mask)]:
                                rejected_response = responses[random.choice(np.flatnonzero(~mask))]

                                new["chosen"].append(chosen_response)
                                new["rejected"].append(rejected_response)
                                new["prompt"].append(prompt)
                    return new

                dataset = load_dataset("json", data_files=os.path.join(ds, split, "merged.json"), split='train')
                # toy first round of examples, will be updated in collator
                dataset = dataset.map(transform, batched=True, remove_columns=dataset.column_names)

            else:
                dataset = load_dataset("parquet", data_files=os.path.join(ds, "data", f"{split}-*.parquet"), split='train')

            # Remove redundant columns to avoid schema conflicts on load
            dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])
            if "train" in split:
                raw_train_datasets.append(dataset)
            elif ("test" in split) or ("validation" in split):
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")


    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(seed=42)
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with splits {splits}. Check the dataset has been correctly formatted."
        )

    return raw_datasets

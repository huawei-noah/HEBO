# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
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

import math
import numpy as np
import random

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from trl.trainer.utils import DPODataCollatorWithPadding
from typing import Any, Dict, List, Union


# Defined functions
def sft_format(example):

    output_texts = []
    for a in example['answers']:
        output_texts.append(f"### Question: {example['question']}\n ### Answer: {a['text']}")

    return output_texts


def keep_top(example, p, static):

    answers = [a for a in example['answers'] if a['votes'] is not None]
    i = math.ceil(p * len(answers) / 100)
    example['answers'] = [random.choice(answers[:i])] if static else answers[:i]

    return example


def dpo_format(example, task):

    responses, scores = [], []
    for a in example['answers']:
        responses.append(a['text'])
        scores.append(np.inf if a['votes'] is None else float(a['votes']))

    responses, scores = np.array(responses), np.array(scores)
    mask = np.isfinite(scores)

    if task == 'qvs':

        i = random.choice(np.flatnonzero(mask)[:-1])

        chosen_response = responses[i]
        rejected_response = random.choice(responses[i + 1:])

    elif task == 'pvf':

        chosen_response = responses[random.choice(np.flatnonzero(mask))]
        rejected_response = responses[random.choice(np.flatnonzero(~mask))]

    elif task == 'all':

        i = random.choice(np.flatnonzero(mask))

        chosen_response = responses[i]
        rejected_response = random.choice(responses[i + 1:])

    return {'chosen_response': chosen_response, 'rejected_response': rejected_response}


# Defined classes
class SFTTrainer(SFTTrainer):

    def _prepare_non_packed_dataloader(
        self, tokenizer, dataset, dataset_text_field, max_seq_len, formatting_func=None
    ):
        use_formatting_func = formatting_func is not None and dataset_text_field is None
        self._dataset_sanity_checked = False

        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field] if not use_formatting_func else formatting_func(element),
                truncation=True,
                padding=False,
                max_length=max_seq_len,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if use_formatting_func and not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

        tokenized_dataset = dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )

        return tokenized_dataset


class SFTCollator(DataCollatorForCompletionOnlyLM):

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:

        tokenized_batch = []
        for example in examples:

            i = random.choice(range(len(example['input_ids'])))
            batch_element = {'input_ids': example['input_ids'][i], 'attention_mask': example['attention_mask'][i]}
            tokenized_batch.append(batch_element)

        return super().torch_call(tokenized_batch)


class DPOCollator(DPODataCollatorWithPadding):

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        tokenized_batch = []
        for feature in features:

            output = dpo_format(feature, self.task) if 'answers' in feature else feature
            chosen_response, rejected_response = output['chosen_response'], output['rejected_response']            

            batch_element = self.tokenize_batch_element(feature['question'], chosen_response, rejected_response)
            tokenized_batch.append(batch_element)

        # return collated batch
        return self.collate(tokenized_batch)

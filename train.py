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
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import pandas as pd
import warnings

from datasets import disable_caching
disable_caching()

from datasets import Dataset
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers.utils import logging
from trl import DPOTrainer

from utils.collator import sft_format, keep_top, dpo_format
from utils.collator import SFTTrainer, SFTCollator, DPOCollator

logging.set_verbosity_info()
logger = logging.get_logger('transformers')
logger.info('INFO')
logger.warning('WARN')


def main():

    # Define the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_path', type=str, required=True, help='The data path to use.'
    )
    parser.add_argument(
        '--model_path', type=str, required=True, help='The model path to use.'
    )
    parser.add_argument(
        '--optim', type=str, choices=['sft', 'dpo'], help='The optimization to use.'
    )
    parser.add_argument(
        '--top_p', type=int, help='The top-p responses for SFT.'
    )
    parser.add_argument(
        '--task', type=str, choices=['qvs', 'pvf', 'all'], help='The pairwise task for DPO.'
    )
    parser.add_argument(
        '--augment', type=str, choices=['True', 'False'], help='Whether to use data augmentation.'
    )
    parser.add_argument(
        '--ds_config', type=str, required=True, help='The deepspeed config to use.'
    )
    parser.add_argument(
        '--save_path', type=str, required=True, help='The save path to use.'
    )
    parser.add_argument(
        '--seed', type=int, required=True, help='The seed to use.'
    )

    args = parser.parse_args()

    # Set the hyperparameters
    DATA_PATH = args.data_path
    MODEL_PATH = args.model_path
    OPTIM = args.optim
    TOP_P = args.top_p
    TASK = args.task
    AUGMENT = args.augment == 'True'
    DS_CONFIG = args.ds_config
    SAVE_PATH = args.save_path
    SEED = args.seed


    # Define the arguments
    train_args = TrainingArguments(
        output_dir=SAVE_PATH,
        evaluation_strategy='epoch',
        prediction_loss_only=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=5e-7,
        lr_scheduler_type='linear',
        warmup_ratio=0.1,
        logging_strategy='epoch',
        num_train_epochs=30,
        save_strategy='epoch',
        save_total_limit=2,
        fp16=True,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        deepspeed=DS_CONFIG,
        report_to=['tensorboard'],
        gradient_checkpointing=True
    )

    # Set the seed
    set_seed(SEED)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model.name_or_path,
        model_max_length=2048,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'


    # Load the dataset
    train_data = pd.read_json(os.path.join(DATA_PATH, 'train', 'merged.json'), orient='records', lines=True)
    train_data = Dataset.from_pandas(train_data)

    val_data = pd.read_json(os.path.join(DATA_PATH, 'validation', 'merged.json'), orient='records', lines=True)
    val_data = Dataset.from_pandas(val_data)

    if OPTIM == 'sft':
        train_data = train_data.map(keep_top, fn_kwargs={'p': TOP_P, 'static': False if AUGMENT else True})  
        val_data = val_data.map(keep_top, fn_kwargs={'p': TOP_P, 'static': True})

        response_template = tokenizer.encode(' ### Answer:', add_special_tokens=False)
        if model.config.model_type == 'llama':
            response_template = response_template[1:]

        data_collator = SFTCollator(response_template, tokenizer=tokenizer)

    else:
        if not AUGMENT:
            train_data = train_data.map(dpo_format, remove_columns=['answers'], fn_kwargs={'task': TASK})

        val_data = val_data.map(dpo_format, remove_columns=['answers'], fn_kwargs={'task': TASK})

        data_collator = DPOCollator(
            tokenizer=tokenizer, 
            max_length=tokenizer.model_max_length, 
            max_prompt_length=tokenizer.model_max_length // 2, 
            max_target_length=tokenizer.model_max_length // 2
        )
        data_collator.task = TASK

    data_module = dict(
        data_collator=data_collator,
        train_dataset=train_data, 
        eval_dataset=val_data
    )
    if OPTIM == 'sft':
        data_module['formatting_func'] = sft_format
        data_module['max_seq_length'] = tokenizer.model_max_length
    else:
        data_module['ref_model'] = model.name_or_path

    # Train the model
    trainer = {'sft': SFTTrainer, 'dpo': DPOTrainer}[OPTIM]
    trainer = trainer(model=model, tokenizer=tokenizer, args=train_args, **data_module)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        trainer.train()


if __name__ == '__main__':
    main()

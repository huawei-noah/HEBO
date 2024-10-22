#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
# https://github.com/huggingface/alignment-handbook/blob/main/scripts/run_dpo.py

import logging
import random
import sys
import os
import torch
import transformers
import datasets
import re
from transformers import AutoModelForCausalLM, AutoConfig, set_seed, TrainingArguments
import wandb

from src.alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    decontaminate_humaneval,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from transformers.trainer_utils import get_last_checkpoint

from peft import PeftConfig, PeftModel
from src.trainers import (
    MAPOConfig, MAPOTrainer,
    SparseConfig, SparseTrainer,
)
from src.alignment.utils import get_mapo_model, get_sparse_pipeline

logger = logging.getLogger(__name__)


def main():
    if '--pref_optim=mapo' in sys.argv:
        parser = H4ArgumentParser((ModelArguments, DataArguments, MAPOConfig))
    elif '--pref_optim=sparse' in sys.argv:
        parser = H4ArgumentParser((ModelArguments, DataArguments, SparseConfig))
    else:
        raise ValueError('Invalid preference optimization flag!')

    model_args, data_args, training_args = parser.parse()

    training_args.logging_dir = os.path.join(training_args.output_dir, 'runs', training_args.logging_dir.split('/')[-1])

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    if 'mbpp_new' not in list(data_args.dataset_mixer.keys())[0]:
        data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
        tokenizer = get_tokenizer(model_args, data_args)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'  # to prevent errors with FA
        tokenizer.truncation_side = 'left'  # to prevent cutting off last generation
        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    else:
        data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
        tokenizer = get_tokenizer(model_args, data_args)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'  # to prevent errors with FA
        tokenizer.truncation_side = 'left'  # to prevent cutting off last generation
        tokenizer.bos_token = None
        tokenizer.bos_token_id = None

    ###############
    # Load datasets
    ###############
    def filter_dataset(examples):
        query = examples['prompt']
        prompt_length = tokenizer.apply_chat_template([{'content': query, 'role': 'user'}], tokenize=True, add_generation_prompt=True, return_tensors='pt').size(-1)

        if model_args.pref_optim == 'sft':
            return prompt_length < 1024
        else:
            return all([prompt_length < 1024,
                        examples['chosen'] != examples['rejected'],
                        examples['chosen'][-1]['content'] != "",
                        examples['rejected'][-1]['content'] != "", ])
            

    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"],
    )
    logger.info(raw_datasets)

    # Filter based on prompt length
    num_raw_train_samples = len(raw_datasets["train"])
 
    if 'mbpp_new' not in list(data_args.dataset_mixer.keys())[0]:
        raw_datasets = raw_datasets.filter(filter_dataset)

        num_filtered_train_samples = num_raw_train_samples - len(raw_datasets["train"])
        logger.info(
            f"Filtered based on prompt length + none {num_filtered_train_samples} ({num_filtered_train_samples/num_raw_train_samples * 100:.2f}%) samples from the training set."
        )
        logger.info(raw_datasets)

    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################
    # Apply chat template
    #####################
    if 'mbpp_new' not in list(data_args.dataset_mixer.keys())[0]:
        raw_datasets = raw_datasets.map(
            apply_chat_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "task": model_args.pref_optim,
                "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            },
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Formatting comparisons with prompt template",
        )
    else:
        raw_datasets = raw_datasets.rename_column('prompt', "text_prompt")
        raw_datasets = raw_datasets.rename_column('chosen', "text_chosen")
        raw_datasets = raw_datasets.rename_column('rejected', "text_rejected")

    ##########################
    # Decontaminate benchmarks
    ##########################
    num_raw_train_samples = len(raw_datasets["train"])
    raw_datasets['train'] = raw_datasets['train'].filter(
        decontaminate_humaneval,
        fn_kwargs={"text_column": "text_chosen"} if model_args.pref_optim != 'sft' else {"text_column": "text"},
        batched=True,
        batch_size=10_000,
        num_proc=1,
        desc="Decontaminating HumanEval samples",
    )
    num_filtered_train_samples = num_raw_train_samples - len(raw_datasets["train"])
    logger.info(
        f"Decontaminated {num_filtered_train_samples} ({num_filtered_train_samples/num_raw_train_samples * 100:.2f}%) samples from the training set."
    )
    logger.info(raw_datasets)

    if model_args.pref_optim != "sft":
        # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
        raw_datasets = raw_datasets.rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

        # Log a few random samples from the training set:
        for index in random.sample(range(len(raw_datasets["train"])), 3):
            logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
            logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
            logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")

    else:
        for index in random.sample(range(len(raw_datasets["train"])), 3):
            logger.info(f"Sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['text']}")

    ########
    # MODEL
    ########
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = None
    if model_args.pref_optim == "sparse":
        model = get_sparse_pipeline(model_args.model_name_or_path, config, training_args, model_kwargs)
        
    elif model_args.pref_optim == "mapo":
        model = get_mapo_model(model_args.model_name_or_path, config.architectures, model_kwargs)

    peft_config = get_peft_config(model_args)
    ref_model = None
    if peft_config is None:
        if model_args.pref_optim != "mapo":
            ref_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
        else:
            ref_model = get_mapo_model(model_args.model_name_or_path, config.architectures, model_kwargs)
        ref_model_kwargs = model_kwargs
    else:
        logger.info(peft_config)
        
        ref_model = None
        ref_model_kwargs = None
    

    #########################
    # Instantiate DPO trainer
    #########################
    logger.info(model)


    if model_args.pref_optim == "mapo":
        logger.info(f'********** Using {model_args.pref_optim.upper()} loss with beta = {training_args.beta} **********')

        trainer = MAPOTrainer(
            model,
            ref_model,
            args=training_args,
            train_dataset=raw_datasets["train"],
            eval_dataset=raw_datasets["test"],
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_args),
        )

    elif model_args.pref_optim == "sparse":
        logger.info(f'********** Using {model_args.pref_optim.upper()} loss beta = {training_args.beta} **********')

        trainer = SparseTrainer(
            model,
            ref_model,
            args=training_args,
            train_dataset=raw_datasets["train"],
            eval_dataset=raw_datasets["test"],
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_args),
        )

    logger.info("[MODEL AFTER TRAINER] check...")
    logger.info(trainer.model)

    ###############
    # Training loop
    ###############
    # Detecting last checkpoint.

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    else:
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        checkpoint = last_checkpoint
    #
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": [model_args.pref_optim],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        metrics = {k:v for k,v in metrics.items() if "_hist" not in k}
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    logger.info("*** Final evaluation complete! ***")


if __name__ == "__main__":
    main()

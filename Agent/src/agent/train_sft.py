# Copyright 2023 Huawei R&D UK.
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import itertools

# Adapted from: https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py
import hydra
import pyrootutils
import torch
from datasets import DatasetDict
from datasets import load_dataset
from omegaconf import DictConfig
from omegaconf import OmegaConf
from peft import PeftConfig
from peft import PeftModel
from peft import get_peft_model
from peft import prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import DataCollatorForCompletionOnlyLM
from trl import SFTTrainer
from trl.trainer.utils import PeftSavingCallback

from agent import utils
from agent.models.huggingface import HuggingFaceBackend
from agent.utils import pylogger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = pylogger.get_pylogger(__name__)

torch.cuda.empty_cache()

DEFAULT_TEXT_FIELD = "text"


def extract_collator_templates_tokens(tokenizer, return_tail=False):
    raise NotImplementedError("Tokenizers can treat spaces inconsistently which makes this function not work well")
    # The apply_chat_template functions should definitely return a user/agent/special token mask for this
    unique_token = tokenizer.unk_token

    chat_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": unique_token}, {"role": "assistant", "content": unique_token}], tokenize=True
    )

    # Split on unk token id
    unk_split = [list(g) for k, g in itertools.groupby(chat_tokens, lambda t: t == tokenizer.unk_token_id) if not k]
    templates = tuple(unk_split)
    log.info("Automatically extracted instruction and response templates: %s", tokenizer.batch_decode(list(templates)))
    if return_tail:
        return templates
    else:
        return templates[:2]


def extract_collator_templates(tokenizer, return_tail=False):
    # The apply_chat_template functions should definitely return a user/agent/special token mask for this
    unique_token = tokenizer.unk_token

    chat_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": unique_token}, {"role": "assistant", "content": unique_token}], tokenize=False
    )
    templates = chat_str.split(" " + unique_token + " ")

    log.info("Automatically extracted instruction and response templates: %s", templates)
    if return_tail:
        return templates
    else:
        return templates[:2]


def prepare_datasets(dataset_cfg, tokenizer, seed=None):
    if isinstance(dataset_cfg.loading_kwargs.get("split"), str):
        # This ensures the load_dataset function is always a dataset dict
        dataset_cfg.train_split = dataset_cfg.loading_kwargs.get("split")
        dataset_cfg.loading_kwargs.split = None
        print("Dataset split specified, overwriting dataset.train_split", dataset_cfg.train_split)

    dataset_dict: DatasetDict = load_dataset(**dataset_cfg.loading_kwargs)

    history_field = dataset_cfg.get("history_field")
    text_field = dataset_cfg.get("text_field", DEFAULT_TEXT_FIELD)

    role_key = dataset_cfg.history.role_key
    content_key = dataset_cfg.history.content_key
    role_mapping = {dataset_cfg.history.assistant_role: "assistant", dataset_cfg.history.user_role: "user"}

    # Check for relevant fields
    for split_name, split_columns in dataset_dict.column_names.items():
        valid_text_field = text_field in split_columns
        valid_history_field = history_field is not None and history_field in split_columns

        if not (valid_text_field or valid_history_field):
            raise ValueError("No valid data field found")
        elif not valid_text_field:
            # Must have a valid history field - generate text field from history
            def text_from_history(example):
                processed_history = []
                last_assistant_i = None
                for i, message in enumerate(example[history_field]):
                    role = role_mapping[message[role_key]]
                    content = message[content_key]
                    if role == "assistant":
                        # Inclusive slice by adding 1
                        last_assistant_i = i + 1
                    # Huggingface expects this format
                    processed_history.append({"role": role, "content": content})

                # Everything after the last assistant utterance can be removed
                # (because it shouldn't affect the likelihood of any agent outputs)
                processed_history = processed_history[:last_assistant_i]

                # tokens = tokenizer.apply_chat_template(processed_history)
                # example[text_field] = tokenizer.decode(tokens, skip_special_tokens=False)
                example[text_field] = tokenizer.apply_chat_template(processed_history, tokenize=False)

                return example

            dataset_dict[split_name] = dataset_dict[split_name].map(text_from_history)

    # TODO: Maybe add a check here for special tokens in the text field
    # (because the text field needs to already be processed like that)
    dataset_train = dataset_dict[dataset_cfg.train_split] if dataset_cfg.get("train_split") is not None else None
    dataset_eval = (
        dataset_dict[dataset_cfg.validation_split] if dataset_cfg.get("validation_split") is not None else None
    )

    if dataset_train is not None:
        dataset_train = dataset_train.shuffle(seed)

    return (dataset_train, dataset_eval), text_field


def configure_peft_training(model, peft_cfg, use_gradient_checkpointing):
    peft_loaded = issubclass(type(model), PeftModel)
    use_peft = peft_loaded or peft_cfg is not None

    if peft_loaded:
        if peft_cfg is not None:
            raise ValueError(
                "A peft config should not be defined in both the pretrained model and the training config",
                model,
                peft_cfg,
            )
        base_model = model.get_base_model()
    else:
        base_model = model

    if use_peft:
        # LoRA with kbit quantization needs to be handled specially
        if getattr(base_model, "is_loaded_in_8bit", False) or getattr(base_model, "is_loaded_in_4bit", False):
            prepare_model_for_kbit_training(
                base_model,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
        else:
            base_model.enable_input_require_grads()

        if peft_cfg is None:
            log.info("No PEFT config found -- training without PEFT")
        else:
            peft_config: PeftConfig = hydra.utils.instantiate(peft_cfg, _convert_="object")
            peft_config.task_type = "CAUSAL_LM"

            return get_peft_model(model, peft_config)

    return model


@hydra.main(version_base="1.3", config_path="../../configs", config_name="default_sa_train_sft.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)
    OmegaConf.resolve(cfg)

    log.info("Starting LLM supervised finetuning")

    # Load model and tokenizer u
    llm: HuggingFaceBackend = hydra.utils.instantiate(cfg.llm, _recursive_=True)
    llm.model = configure_peft_training(llm.model, cfg.get("peft"), cfg.training.gradient_checkpointing)

    # Configure datasets
    (dataset_train, dataset_eval), text_field = prepare_datasets(cfg.dataset, llm.tokenizer, llm.seed)
    # print("max sequence length:", max(len(tokenizer(e[text_field])["input_ids"]) for e in dataset_train))

    # Configure a data collator to mask user tokens from the loss
    instruction_template = cfg.get("instruction_template")
    response_template = cfg.get("response_template")
    if instruction_template is None or response_template is None:
        instruction_template, response_template = extract_collator_templates(llm.tokenizer, return_tail=False)
    data_collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=llm.tokenizer,
    )

    # # Step 5: Define the Trainer
    training_args = TrainingArguments(**cfg.training, **cfg.output)
    callbacks = [PeftSavingCallback] if isinstance(llm.model, PeftModel) else None
    trainer = SFTTrainer(
        model=llm.model,
        tokenizer=llm.tokenizer,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        dataset_text_field=text_field,
        data_collator=data_collator,
        callbacks=callbacks,
        max_seq_length=llm.tokenizer.model_max_length,
    )

    if dataset_train is not None:
        trainer.train()
        # Step 6: Save the model
        trainer.save_model()
    elif dataset_eval is not None:
        print(trainer.evaluate())

    # merged_model = model.merge_and_unload(progressbar=True)
    # merged_model._hf_peft_config_loaded = False
    # merged_model.save_pretrained(script_args.output_dir)
    # tokenizer.save_pretrained(script_args.output_dir) # Optionally include tokenizer in merged model


if __name__ == "__main__":
    main()

import json
import os

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig
from peft import PeftModel
from peft import get_peft_model
from transformers import TrainerCallback
from trl import DataCollatorForCompletionOnlyLM
from trl import SFTTrainer

# from rich import print

torch.cuda.empty_cache()

os.environ["WANDB_PROJECT"] = "agent"
os.environ["WANDB_ENTITY"] = "hf-work"


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if args.should_save:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            kwargs["model"].save_pretrained(checkpoint_path)
            # kwargs["model"].merge_and_unload(progressbar=True).save_pretrained(
            #     checkpoint_path
            # )
            kwargs["tokenizer"].save_pretrained(checkpoint_path)

            if "pytorch_model.bin" in os.listdir(checkpoint_path):
                os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


def build_dataset(tokenizer):
    ddata = {"text": []}
    datasets_paths = os.listdir("../datasets")
    for dp in datasets_paths:
        print("../datasets/" + dp)
        with open("../datasets/" + dp) as file:
            data = list(file)
        dd = {}
        print(data[0])
        for messages in data:
            if dp.startswith("babyai_"):
                task_id = messages["task_id"]
                if task_id in dd.keys():
                    dd[task_id] += 1
                else:
                    dd[task_id] = 1
                if dd[task_id] > 2000:
                    continue
            if "messages" in messages:
                messages = json.loads(messages)["messages"]
            new_messages = {msg["role"]: msg["content"] for msg in messages}
            new = [
                {"role": "user", "content": new_messages["user"]},
                {"role": "assistant", "content": new_messages["assistant"]},
            ]
            text = new_messages["system"] + "<|end_of_turn|>" + tokenizer.apply_chat_template(new, tokenize=False)[3:]
            ddata["text"].append(text)
            # print(text)
        print(dd)
        print(len(ddata["text"]))
        print(repr(ddata["text"][0]))
    dataset = Dataset.from_dict(ddata).train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"].shuffle(seed=42)
    eval_dataset = dataset["test"]

    return dataset, train_dataset, eval_dataset


def main():
    # base_model_id = "meta-llama/Llama-2-7b-chat-hf"
    base_model_id = "openchat/openchat_3.5"
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_id, padding_side="right")

    dataset, train_dataset, eval_dataset = build_dataset(tokenizer)

    local_rank = 0
    if os.environ.get("LOCAL_RANK") is not None:
        # distributed setup, accelerator object will be build inside PPOTrainer
        local_rank = int(os.environ.get("LOCAL_RANK"))
        torch.cuda.set_device(local_rank)
        device_map = f"cuda:{local_rank}"
    else:
        device_map = "balanced"

    # dataset_id = "gsm8k", "main"
    # data = pd.read_json(path_or_buf="converted_dataset.json", lines=False)

    response_template = "GPT4 Correct Assistant:"
    # response_template = "[/INST]"
    response_template_tokenized = tokenizer.encode(f"{response_template}", add_special_tokens=False)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_tokenized, tokenizer=tokenizer)

    example = dataset["train"][0]["text"]
    tokenizer(example)
    # print(collator([example_encoded]))

    # print(response_template_tokenized)
    # print(tokenizer.decode(response_template_tokenized))
    # pipe = pipeline("text-generation", model=base_model_id)

    # print(pipe("hello!!!"))

    peft_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # bnb_config = transformers.BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=bfloat16,
    # )

    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # load_on_init=False,
        # quantization_config=bnb_config,
        device_map=device_map,
        # use_auth_token=hf_auth
    )
    # base_model.gradient_checkpointing_enable()

    # get_peft_model(base_model, peft_config).print_trainable_parameters()

    # print(base_model)

    # dataset = load_dataset(*dataset_id, split="train")

    # def formatting_prompts_func(example):
    #     output_texts = []
    #     for i in range(len(example["question"])):
    #         text = f"### Question: {example['question'][i]}\n ### Answer: {example['answer'][i]}"
    #         output_texts.append(text)
    #     return output_texts
    training_args = transformers.TrainingArguments(
        **{
            # "deepspeed": "deepspeed-config.json",
            # "output_dir": "output",
            "evaluation_strategy": "steps",
            "eval_steps": 0.1,
            "logging_steps": 5,
            "logging_first_step": True,
            "learning_rate": 1e-04,  # LoRA benefits from a relatively high learning rate
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": 10,
            # "dataloader_num_workers": 6,
            # "bf16": True,
            "save_strategy": "steps",
            "save_steps": 0.1,
            "save_total_limit": 10,
            "load_best_model_at_end": True,
            "output_dir": "sft_output2",
            # "tf32": True,
            # "max_steps": 800,
            "gradient_checkpointing": True,
            "ddp_find_unused_parameters": False,  # This avoids errors with lora+gradient_checkpointing+ddp
        }
    )

    base_model.enable_input_require_grads()
    base_model = get_peft_model(base_model, peft_config)
    base_model.config.use_cache = False

    callbacks = [PeftSavingCallback] if isinstance(base_model, PeftModel) else None
    # callbacks = None

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # formatting_func=formatting_prompts_func,
        # peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=4096,
        args=training_args,
        data_collator=collator,
        callbacks=callbacks,
    )
    # print(trainer.evaluate())
    trainer.train()
    # merged_model = base_model.merge_and_unload()
    # merged_model.save_pretrained("checkpoint-11", save_adapter=True, save_config=True)
    # tokenizer.save_pretrained("checkpoint-11")


# print(model)
# dataset = load_dataset("gsm8k", "main")
# print(dataset["train"][100])
if __name__ == "__main__":
    main()

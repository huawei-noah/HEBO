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

import json
import os
import time

import hydra
import shortuuid
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer

from moa_spec.utils import set_and_print_config, set_seed

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)


def load_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions


@torch.inference_mode()
def get_model_answers(
        model,
        tokenizer,
        questions,
        answer_file,
):
    question = questions[0]

    # warmup
    for _ in range(3):
        messages = []
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            messages.append({
                "role": "user",
                "content": qs
            })
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer([prompt],add_special_tokens=False,).input_ids

            model.call_to_big = 0
            torch.cuda.synchronize()
            start_time = time.time()

            output_ids = model.generate(
                torch.as_tensor(input_ids).cuda(),
                do_sample=False,
                max_new_tokens=min(512, 1976 - len(input_ids[0])),
                pad_token_id=128010,
                eos_token_id=[128001, 128009]
            )
            idx = model.call_to_big - 1
            new_token = output_ids.shape[1] - len(input_ids[0])

            torch.cuda.synchronize()
            total_time = time.time() - start_time
            output_ids = output_ids[0][len(input_ids[0]):]
            # be consistent with the template's stop_token_ids
            stop_token_ids = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            if stop_token_ids:
                stop_token_ids_index = [
                    i
                    for i, id in enumerate(output_ids)
                    if id in stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )

            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            messages.append({
                "role": "assistant",
                "content": output
            })

    for question in tqdm(questions):

        choices = []
        for i in range(1):
            messages = []
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                messages.append({
                    "role": "user",
                    "content": qs
                })
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                input_ids = tokenizer([prompt], add_special_tokens=False, ).input_ids

                model.call_to_big = 0
                torch.cuda.synchronize()
                start_time = time.time()

                output_ids = model.generate(
                    torch.as_tensor(input_ids).cuda(),
                    do_sample=False,
                    max_new_tokens=min(512, 1976 - len(input_ids[0])),
                    pad_token_id=128010,
                    eos_token_id=[128001, 128009]
                )
                idx = model.call_to_big - 1
                new_token = output_ids.shape[1] - len(input_ids[0])

                torch.cuda.synchronize()
                total_time = time.time() - start_time
                output_ids = output_ids[0][len(input_ids[0]):]
                # be consistent with the template's stop_token_ids
                stop_token_ids = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                if stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )

                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                messages.append({
                    "role": "assistant",
                    "content": output
                })

            choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")

    total_time = 0
    total_new_tokens = 0
    with open(os.path.expanduser(answer_file), "r") as fin:
        for line in fin.readlines():
            data = json.loads(line)
            total_time += sum(sum(c["wall_time"]) for c in data["choices"])
            total_new_tokens += sum(sum(c["new_tokens"]) for c in data["choices"])

    print(f"Total time: {total_time}, "
          f"Total tokens: {total_new_tokens}, "
          f"Tokens per sec. : {total_new_tokens / total_time}")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


@hydra.main(version_base="1.3", config_path="../configs/inference_single_device", config_name="default_inference")
def main(cfg: DictConfig) -> None:
    set_and_print_config(cfg)

    set_seed(cfg.seed)

    model_kwargs = hydra.utils.instantiate(cfg.model_kwargs)
    model_class = hydra.utils.instantiate(cfg.method.model_class)

    model_config = hydra.utils.instantiate(cfg.method.model_config) if hasattr(cfg.method, "model_config") else {}

    model = model_class.from_pretrained(
        **model_kwargs,
        **model_config,
    )
    if hasattr(model, "custom_load"):
        model.custom_load(cfg.drafter, dtype=model_kwargs["torch_dtype"])
    model.eval()

    tokenizer_kwargs = hydra.utils.instantiate(cfg.tokenizer_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    for bench_name in ["alpaca", "gsm8k", "humaneval", "mt_bench", "qa", "sum"]:
        question_file = f"{parent_dir}/data/{bench_name}/question.jsonl"
        answer_file = f"{output_dir}/{bench_name}.jsonl"

        print(f"reading {question_file}")
        questions = load_questions(question_file)
        get_model_answers(
            model,
            tokenizer,
            questions,
            answer_file,
        )

        reorg_answer_file(answer_file)


if __name__ == "__main__":
    main()

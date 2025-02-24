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
import deepspeed
import json
import pandas as pd
import torch

from datasets import load_from_disk
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from utils.evaluation import evaluate_functional_correctness


def main():

    # Define the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--eval_mode', type=str, choices=['True', 'False'], required=True, help='Whether to evaluate or sample.'
    )
    parser.add_argument(
        '--regen', type=str, choices=['True', 'False'], required=True, help='Whether to regenerate the outputs.'
    )
    parser.add_argument(
        '--data_path', type=str, required=True, help='The data path to use.'
    )
    parser.add_argument(
        '--model_path', type=str, help='The model path to use.'
    )
    parser.add_argument(
        '--n_seq', type=int, help='The number of independently computed returned sequences.'
    )
    parser.add_argument(
        '--n_iter', type=int, help='The number of iterations for sharding the inference.'
    )
    parser.add_argument(
        '--sample', type=str, choices=['True', 'False'], help='Whether or not to use sampling.'
    )
    parser.add_argument(
        '--temp', type=float, help='The value used to modulate the next token probabilities.'
    )
    parser.add_argument(
        '--save_path', type=str, required=True, help='The save path to use.'
    )
    parser.add_argument(
        '--seed', type=int, required=True, help='The seed to use.'
    )

    args = parser.parse_args()

    # Set the hyperparameters
    EVAL_MODE = args.eval_mode == 'True'
    REGEN = args.regen == 'True'
    DATA_PATH = args.data_path
    MODEL_PATH = args.model_path
    N_SEQ = args.n_seq
    N_ITER = args.n_iter
    SAMPLE = args.sample == 'True'
    TEMP = args.temp
    SAVE_PATH = args.save_path
    SEED = args.seed

    # Check the hyperparameters
    if REGEN and N_SEQ % N_ITER !=0:
        raise ValueError('The n_seq must be divisible by the n_iter.')


    # Set the seed
    set_seed(SEED)

    if REGEN:

        # Load the model
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

        ds_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.float16,
            mp_size=torch.cuda.device_count(),
            replace_with_kernel_inject=False
        )
        model = ds_engine.module

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id


    # Load the dataset
    data = load_from_disk(DATA_PATH)
    splits = ['test'] if EVAL_MODE else ['train', 'validation']

    for split in splits:
        problems = data[split].to_pandas()
        problems = problems.set_index('task_id', drop=False)
        problems = problems.to_dict('index')

        save_path = SAVE_PATH if EVAL_MODE else os.path.join(SAVE_PATH, split)

        if REGEN:
            try:
                os.makedirs(save_path)
            except:
                pass

            samples = []
            max_len = 512
            for problem in tqdm(problems.values()):

                with torch.no_grad():
                    prompt = problem['prompt'].strip()
                    inputs = tokenizer.encode(prompt, return_tensors='pt')
                    inputs = inputs.to(model.device)

                    if inputs.shape[1] > max_len:
                        for _ in range(N_SEQ):
                            samples.append(dict(task_id=problem['task_id'], completion=''))
                        continue

                    # Generate the samples
                    for _ in range(N_ITER):
                        outputs = model.generate(
                            input_ids=inputs,
                            max_length=max_len,
                            do_sample=SAMPLE,
                            temperature=TEMP,
                            num_return_sequences=N_SEQ // N_ITER,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id 
                        )

                        for output in outputs:
                            completion = output[inputs.shape[1]:]
                            response = tokenizer.decode(completion)
                            
                            if response.find('def ') != -1:
                                response = response[:response.find('def ')]
                            response = response[:response.find(tokenizer.pad_token)]
                            
                            samples.append(dict(task_id=problem['task_id'], completion=response))

            with open(os.path.join(save_path, 'outputs.json'), 'w') as f:
                json.dump(samples, f)

        else:
            with open(os.path.join(save_path, 'outputs.json'), 'r') as f:
                samples = json.load(f)

            subfolder = os.path.join(save_path, 'runs')
            if not os.path.isdir(subfolder):
                os.makedirs(subfolder)

            for i in range(5):
                print(f'\nCheck {i}:\n')

                file = os.path.join(subfolder, f'check_{i}.json')
                if os.path.isfile(file):
                    print('Skipping as results already exists.')
                    continue

                results = evaluate_functional_correctness(samples, problems, subfolder)
                results = pd.DataFrame([r[1] for result in results.values() for r in result])
                results.to_json(file, orient='records', lines=True)


if __name__ == '__main__':
    main()

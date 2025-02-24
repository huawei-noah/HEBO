# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
# Copyright 2021 OpenAI. All rights reserved.
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

import itertools
import json
import numpy as np
import os

from collections import defaultdict
from collections import Counter
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import List, Union

from .execution import check_correctness


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def evaluate_functional_correctness(
    samples: list,
    problems: dict,
    save_path: str,
    k: List[int] = [1, 10, 100],
    n_workers: int = 1,
    timeout: float = 3.0
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    # Check the generated samples against test suites
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0

        for sample in samples:
            task_id = sample['task_id']
            completion = sample['completion']
            
            args = (problems[task_id], completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            
            completion_id[task_id] += 1
            n_samples += 1

        assert len(completion_id) == len(problems), 'Some problems are not attempted.'

        results = defaultdict(list)

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result['task_id']].append((result['completion_id'], result))

    # Calculate metrics
    total, correct, speed = [], [], []

    for result in results.values():

        passed = []
        result.sort()

        for r in result:
            passed.append(r[1]['output'] == 'passed')
            if passed[-1] == True:
                speed.append(r[1]['avg_time'])

        total.append(len(passed))
        correct.append(sum(passed))

    total = np.array(total)
    correct = np.array(correct)
    speed = np.array(speed)

    metrics = {f'pass@{n}': estimate_pass_at_k(total, correct, n).mean() for n in k if (total >= n).all()}
    metrics['avg_time'] = np.mean(speed)
    print(metrics)

    out_file = os.path.join(save_path, 'results.txt')
    with open(out_file, 'w' if not os.path.isfile(out_file) else 'a') as f:
        f.write(json.dumps(metrics) + '\n')

    return results

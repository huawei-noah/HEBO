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
import argparse
import numpy as np
import os
import pandas as pd


def main():

    # Define the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_path', type=str, required=True, help='The data path to use.'
    )
    parser.add_argument(
        '--save_path', type=str, required=True, help='The save path to use.'
    )

    args = parser.parse_args()

    # Set the hyperparameters
    DATA_PATH = args.data_path
    SAVE_PATH = args.save_path


    # Load the samples
    for split in sorted(os.listdir(DATA_PATH)):

        results = None
        subfolder = os.path.join(DATA_PATH, split, 'runs')

        for file in os.listdir(subfolder):
            if file.endswith('.json'):
                dataset = pd.read_json(os.path.join(subfolder, file), orient='records', lines=True)
                results = dataset if results is None else pd.concat([results, dataset], ignore_index=True)

        results = results.groupby(['task_id', 'completion_id', 'prompt', 'completion', 'output'], as_index=False, sort=False)
        results = results[['avg_time', 'std_time']].mean()

        # Filter the samples
        total_prob = len(results['task_id'].unique())
        total_sol = len(results)

        results['votes'] = (results['output'] == 'passed').astype(int) * results['avg_time']
        results['votes'].fillna(np.inf, inplace=True)
        results = results.sort_values(['task_id', 'votes'])

        results = results.drop_duplicates(subset=['completion'])
        mask = results.groupby('task_id', sort=False)['votes'].transform(lambda x : sum(x != np.inf) >= 2 and sum(x == np.inf) >= 1)
        results = results[mask]

        filtered_prob = len(results['task_id'].unique())
        filtered_sol = len(results)

        ratio_prob = round(filtered_prob / total_prob * 100, 2)
        ratio_sol = round(filtered_sol / total_sol * 100, 2)

        passed_count = results[results['votes'] != np.inf].groupby('task_id', sort=False)['completion'].count()

        avg_time = results.groupby('task_id', sort=False)['avg_time'].mean()
        std_time = results.groupby('task_id', sort=False)['avg_time'].std()

        results = results.rename(columns={'task_id': 'id', 'prompt': 'question', 'completion': 'text'})
        results['answers'] = results[['text', 'votes']].agg(lambda x: dict(zip(x.index, x.values)), axis=1)
        results = results[['id', 'question', 'answers']]
        results = results.groupby(['id', 'question'], sort=False)['answers'].apply(list).reset_index()

        results['count'] = passed_count.values
        results['avg'] = avg_time.values
        results['std'] = std_time.values
        results['cov'] = results['std'] / results['avg']

        results['time'] = results[['count', 'avg', 'std', 'cov']].agg(lambda x: dict(zip(x.index, x.values)), axis=1)
        results = results.drop(['count', 'avg', 'std', 'cov'], axis=1)


        # Save the dataset
        merge_path = os.path.join(SAVE_PATH, split)
        if not os.path.exists(merge_path):
            os.makedirs(merge_path)
        
        results.to_json(os.path.join(merge_path, 'merged.json'), orient='records', lines=True)

        with open(os.path.join(merge_path, 'results.txt'), 'w') as f:
            print(f'Problems (total): {total_prob}', file=f)
            print(f'Problems (filtered): {filtered_prob}', file=f)
            print(f'Problems (ratio): {ratio_prob:.2f}\n', file=f)

            print(f'Solutions (total): {total_sol}', file=f)
            print(f'Solutions (filtered): {filtered_sol}', file=f)
            print(f'Solutions (ratio): {ratio_sol:.2f}', file=f)

        print(f'\n{split.capitalize()}\n')
        with open(os.path.join(merge_path, 'results.txt'), 'r') as f:
            print(f.read())


if __name__ == '__main__':
    main()

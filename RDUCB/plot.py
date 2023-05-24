"""
    MIT License

    Copyright (c) 2023  Huawei Technologies Co., Ltd

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

"""

import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import sys
import json
import argparse

plt.rc('font', size=20) #controls default text size
plt.rc('axes', labelsize=26) #fontsize of the x and y labels
plt.rc('xtick', labelsize=20) #fontsize of the x tick labels
plt.rc('ytick', labelsize=20) #fontsize of the y tick labels
plt.rc('legend', fontsize=15) #fontsize of the legend
plt.figure(figsize=(10, 7.5))

def get_matching_runs(data_type, data_name, exehashs, ignore_error=False, run_length=None):
    data = None

    for name, exehash in exehashs.items():
        run_no = 0
        used_seeds = set()
        for id_ in os.listdir('mlruns/0/'):
            if id_=='meta.yaml':
                continue
            if 'hash_data' not in os.listdir(os.path.join('mlruns/0/', id_, 'tags')) or 'hash_exe' not in os.listdir(os.path.join('mlruns/0/', id_, 'tags')):
                continue

            if type(data_type)!=type(None) and type(data_name)!=type(None):
                if data_type not in os.listdir(os.path.join('mlruns/0/', id_, 'params')):
                    continue
                
                if pd.read_csv(os.path.join('mlruns/0/', id_, 'params', data_type), header=None, sep=' ')[0].iloc[0] != data_name:
                    continue

            if pd.read_csv(os.path.join('mlruns/0/', id_, 'params', 'param_file'), header=None, sep=' ')[0].iloc[0] == exehash:
                early_stopped = False
                if 'Error' in os.listdir(os.path.join('mlruns/0/', id_, 'tags')) and not ignore_error:
                    if pd.read_csv(os.path.join('mlruns/0/', id_, 'tags', 'Error'), header=None, sep=' ')[0].iloc[0] != 'Exception':
                        print('Error in', id_)
                        continue
                    else:
                        early_stopped = True
                try:
                    it = pd.read_csv(os.path.join('mlruns/0/', id_, 'metrics', 'best_regret'), header=None, sep=' ')
                    best_regret = pd.read_csv(os.path.join('mlruns/0/', id_, 'metrics', 'best_regret'), header=None, sep=' ')
                    cum_regret = pd.read_csv(os.path.join('mlruns/0/', id_, 'metrics', 'cum_instant_regret'), header=None, sep=' ')[1]
                    avg_cum_instant_regret = pd.read_csv(os.path.join('mlruns/0/', id_, 'metrics', 'avg_cum_instant_regret'), header=None, sep=' ')[1]
                    inst_regret = pd.read_csv(os.path.join('mlruns/0/', id_, 'metrics', 'instant_regret'), header=None, sep=' ')[1]
                    seed = pd.read_csv(os.path.join('mlruns/0/', id_, 'params', 'algorithm_random_seed'), header=None, sep=' ')[0].iloc[0]
                    f_min = pd.read_csv(os.path.join('mlruns/0/', id_, 'tags', 'f_min'), header=None, sep=' ')[0].iloc[0]
                except:
                    continue

                if max(it[2]) < run_length and not early_stopped:
                    print('Not long enough ', id_)
                    continue
                #print(id_)
                if seed not in used_seeds:
                    used_seeds.add(seed)
                    run_no += 1
                    if not early_stopped:
                        print('Accepted', id_, "Len", max(it[2]))
                    else:
                        print('Early Stopped', id_, "Len", max(it[2]))
                    
                else:
                    continue

                new_data = pd.DataFrame({
                    'Evaluation step': best_regret[2],
                    'Best regret': best_regret[1] - f_min,
                    'seed': seed,
                    'algorithm': name
                })

                if type(data) == type(None):
                    data = new_data
                else:
                    data = pd.concat([data, new_data], ignore_index=True)

        print(f'Found {run_no} runs for {name}')

    return data

parser = argparse.ArgumentParser()
parser.add_argument("runs_file")
parser.add_argument("run_length")
parser.add_argument("--legend", action="store_true")
parser.add_argument("--sci", action="store_true")
args = parser.parse_args()
runs_file = args.runs_file
run_length = int(args.run_length)
legend = args.legend
scientific_notation = args.sci

run_hashes = json.load(open(runs_file, 'rb'))
if 'sub_benchmark' in run_hashes:
    data_type, data_name = 'sub_benchmark', run_hashes.pop('sub_benchmark')
else:
    data_type = None
    data_name = None


data = get_matching_runs(data_type, data_name, run_hashes, run_length=run_length)
if data is None:
    print("No matching runs found!")
else:
    sns.lineplot(data=data[(data["Evaluation step"]>0) & (data["Evaluation step"]<run_length)], x="Evaluation step", y='Best regret', hue='algorithm', style='algorithm', errorbar=("se", 1), legend="auto" if legend else False)
    if scientific_notation:
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    out_filename = "".join(runs_file.split(".")[:-1])
    plt.savefig(f'{out_filename}.png')

    print(f"Saving results in {out_filename}.png")
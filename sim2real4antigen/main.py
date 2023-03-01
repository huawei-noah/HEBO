import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import importlib
import pickle as pkl

ROOT = str(Path(os.path.realpath(__file__)).parent)
# # ========== Read original data and remove duplicates ==========
# data = pd.read_csv(os.path.join(ROOT, 'data/her2Absolut.tsv'), sep='\t')
# is_dup_mask = data.duplicated()
# data = data[~is_dup_mask]
# data.to_csv(os.path.join(ROOT, 'data/her2Absolut_nodup.csv'), sep=',', index=False)
# data.to_csv(os.path.join(ROOT, 'data/her2Absolut_nodup.tsv'), sep='\t', index=False)
data = pd.read_csv(os.path.join(ROOT, 'data/her2Absolut_nodup.csv'))

# ========== Get threshold ==========
energies = data['Energy'].values
sorted_energies = sorted(energies)
top002 = sorted_energies[:int(0.002 * len(sorted_energies))]
np.save(os.path.join(ROOT, 'data/her2Absolut_nodup/top002_energy_theshold.npy'), np.array(top002[-1]))
top0035 = sorted_energies[:int(0.0035 * len(sorted_energies))]
np.save(os.path.join(ROOT, 'data/her2Absolut_nodup/top0035_energy_theshold.npy'), np.array(top0035[-1]))

# ========== Create Train/Test sets for TransVAE ==========
mols_train = data["Slide"].iloc[:int(0.8 * len(data))]
mols_test = data["Slide"].iloc[int(0.8 * len(data)):]
props_train = data["Energy"].iloc[:int(0.2 * len(data))]
props_test = data["Energy"].iloc[int(0.2 * len(data)):]
props_train_std = (props_train - props_train.mean()) / props_train.std()
props_test_std = (props_test - props_test.mean()) / props_test.std()

sorted_props_idx = data["Energy"].values.flatten().argsort()
top_1perc_idx = sorted_props_idx[:250_000]
rand_1rest_idx = np.random.choice(sorted_props_idx[250_000:], size=250_000, replace=False)
top1_and_1rand_idx = np.concatenate((top_1perc_idx, rand_1rest_idx))
np.random.shuffle(top1_and_1rand_idx)  # shuffle so there is good and bad points in both test and train
top1_and_1rand_mols = data["Slide"].iloc[top1_and_1rand_idx]
top1_and_1rand_mols_train = top1_and_1rand_mols[:int(0.8 * len(top1_and_1rand_mols))]
top1_and_1rand_mols_test = top1_and_1rand_mols[int(0.8 * len(top1_and_1rand_mols)):]
top1_and_1rand_props = data["Energy"].iloc[top1_and_1rand_idx]
top1_and_1rand_props_std = (top1_and_1rand_props - data["Energy"].mean()) / data["Energy"].std()
top1_and_1rand_props_std_train = top1_and_1rand_props_std[:int(0.8 * len(top1_and_1rand_mols))]
top1_and_1rand_props_std_test = top1_and_1rand_props_std[int(0.8 * len(top1_and_1rand_mols)):]

# os.makedirs(os.path.join(ROOT, 'data/her2Absolut_nodup'), exist_ok=True)
# mols_train.to_csv(os.path.join(ROOT, 'data/her2Absolut_nodup/mols_train.csv'), sep=',', index=False)
# mols_test.to_csv(os.path.join(ROOT, 'data/her2Absolut_nodup/mols_test.csv'), sep=',', index=False)
# props_train.to_csv(os.path.join(ROOT, 'data/her2Absolut_nodup/props_train.csv'), sep=',', index=False)
# props_test.to_csv(os.path.join(ROOT, 'data/her2Absolut_nodup/props_test.csv'), sep=',', index=False)
# props_train_std.to_csv(os.path.join(ROOT, 'data/her2Absolut_nodup/props_train_std.csv'), sep=',', index=False)
# props_test_std.to_csv(os.path.join(ROOT, 'data/her2Absolut_nodup/props_test_std.csv'), sep=',', index=False)
top1_and_1rand_mols_train.to_csv(os.path.join(ROOT, 'data/her2Absolut_nodup/top1_and_1rand_mols_train.csv'), sep=',', index=False)
top1_and_1rand_mols_test.to_csv(os.path.join(ROOT, 'data/her2Absolut_nodup/top1_and_1rand_mols_test.csv'), sep=',', index=False)
top1_and_1rand_props_std_train.to_csv(os.path.join(ROOT, 'data/her2Absolut_nodup/top1_and_1rand_props_std_train.csv'), sep=',', index=False)
top1_and_1rand_props_std_test.to_csv(os.path.join(ROOT, 'data/her2Absolut_nodup/top1_and_1rand_props_std_test.csv'), sep=',', index=False)

# idx = np.random.choice(np.arange(len(data)), size=100_000, replace=False)
# train_idx = idx[:80_000]
# test_idx = idx[80_000:]
# mols_train_small = data["Slide"].iloc[train_idx]
# mols_test_small = data["Slide"].iloc[test_idx]
# props_train_small = data["Energy"].iloc[train_idx]
# props_test_small = data["Energy"].iloc[test_idx]
# props_train_std_small = (props_train_small - props_train.mean()) / props_train.std()
# props_test_std_small = (props_test_small - props_test.mean()) / props_test.std()
# # os.makedirs(os.path.join(ROOT, 'data/her2Absolut_nodup'), exist_ok=False)
# mols_train_small.to_csv(os.path.join(ROOT, 'data/her2Absolut_nodup/mols_train_small.csv'), sep=',', index=False)
# mols_test_small.to_csv(os.path.join(ROOT, 'data/her2Absolut_nodup/mols_test_small.csv'), sep=',', index=False)
# props_train_small.to_csv(os.path.join(ROOT, 'data/her2Absolut_nodup/props_train_small.csv'), sep=',', index=False)
# props_test_small.to_csv(os.path.join(ROOT, 'data/her2Absolut_nodup/props_test_small.csv'), sep=',', index=False)
# props_train_std_small.to_csv(os.path.join(ROOT, 'data/her2Absolut_nodup/props_train_std_small.csv'), sep=',', index=False)
# props_test_std_small.to_csv(os.path.join(ROOT, 'data/her2Absolut_nodup/props_test_std_small.csv'), sep=',', index=False)
#
# # ========== Create Vocabulary for TransVAE (takes a bit of time) ==========
# # data = data.iloc[:100000]
# strings = data['Slide'].str.split("", 11, expand=True)
# vocab = set()
# for c in strings.columns[1:]:
#     print(c)
#     vocab.symmetric_difference_update(set(strings[c].values))
# vocab = sorted(vocab)
# vocab = ['<start>'] + vocab + ['_', '<end>']
# # vocab = ['<start>'] + vocab + ['<end>']
# char_dict = {char: i for i, char in enumerate(vocab)}
# pkl.dump(char_dict, open(os.path.join(ROOT, 'data/her2Absolut_nodup/char_dict.pkl'), 'wb'))
# # char_dict_zinc = pkl.load(open(os.path.join(ROOT, "TransVAE/data/char_dict_zinc.pkl"), 'rb'))
char_dict = pkl.load(open(os.path.join(ROOT, "data/her2Absolut_nodup/char_dict.pkl"), 'rb'))
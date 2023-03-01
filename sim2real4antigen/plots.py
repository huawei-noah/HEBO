import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import importlib
import pickle as pkl

# ROOT = str(Path(os.path.realpath(__file__)).parent)
# all_data = pd.read_csv(os.path.join(ROOT, 'data/her2Absolut_nodup.csv')).to_numpy()
# props_mean, props_std = all_data[:, 1].mean(), all_data[:, 1].std()
# np.save(os.path.join(ROOT, 'data/her2Absolut_nodup/props_mean.npy'), props_mean)
# np.save(os.path.join(ROOT, 'data/her2Absolut_nodup/props_std.npy'), props_std)
#
# energies = np.load('/nfs/aiml/alexandre/Projects/LSBO/data/her2Absolut_nodup/energies.npy')
# mols = pd.read_csv('/nfs/aiml/alexandre/Projects/LSBO/data/her2Absolut_nodup/top1_and_1rand_mols_test.csv').to_numpy()
# labels = pd.read_csv('/nfs/aiml/alexandre/Projects/LSBO/data/her2Absolut_nodup/top1_and_1rand_props_std_test.csv').to_numpy()
# labels = labels * all_data[:, 1].std() + all_data[:, 1].mean()
#
# plt.scatter(labels[:len(energies)], energies, alpha=0.25, marker='.')
# plt.plot(np.linspace(-130, -80, 200), np.linspace(-130, -80, 200), color='k')
# plt.title('Test labels vs. Decoded energies')
# plt.show()
# plt.close()
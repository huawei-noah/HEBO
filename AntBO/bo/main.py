import os
import random
import sys
from pathlib import Path
from typing import Optional, Set, Any, Dict

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path.insert(0, ROOT_PROJECT)

from utilities.misc_utils import log
from bo.custom_init import get_initial_dataset_path, InitialBODataset, get_top_cut_ratio_per_cat, get_n_per_cat
from bo.botask import BOTask as CDRBO
from bo.optimizer import Optimizer
import os
import time
import torch
import pandas as pd
import numpy as np
from bo.utils import save_w_pickle, load_w_pickle
import argparse

SEARCH_STRATEGIES: Set[str] = {'glocal', 'local', 'local-no-hamming', 'batch_local', 'global'}


def get_x_y_from_csv(csv_path):
    data = pd.read_csv(csv_path)
    x = data["x"].values
    from bo.localbo_utils import AA_to_idx
    x = np.array([[AA_to_idx[cat] for cat in xx] for xx in x])
    y = torch.tensor(data["y"].values).reshape(-1, 1)
    return x, y


class BOExperiments:
    def __init__(self, config: Dict[str, Any], cdr_constraints: bool, seed: int):
        """

        :param config: dictionary of parameters for BO
                acq: choice of the acquisition function
                ard: whether to enable automatic relevance determination
                save_path: path to save model and results
                kernel_type: choice of kernel
                normalise: normalise the target for the GP
                batch_size: batch size for BO
                max_iters: maximum evaluations for BO
                n_init: number of initialising random points
                min_cuda: number of initialisation points to use CUDA
                device: default 'cpu' if GPU specify the id
                seq_len: length of seqence for BO
                bbox: dictionary of parameters of blackbox
                    antigen: antigen to use for BO
                seed: random seed

        """
        self.config = config
        self.seed = seed
        self.cdr_constraints = cdr_constraints
        # Sanity checks
        assert self.config['acq'] in ['ucb', 'ei', 'thompson', 'eiucb', 'mace',
                                      'imace'], f"Unknown acquisition function choice {self.config['acq']}"
        if 'search_strategy' in self.config:
            self.search_strategy = self.config['search_strategy']
            assert self.search_strategy in SEARCH_STRATEGIES, print(
                f"{self.search_strategy} not in {SEARCH_STRATEGIES}")
        else:
            self.search_strategy = 'local'

        print(f"Search Strategy {self.search_strategy}")

        self.custom_initial_dataset: Optional[InitialBODataset] = None
        if self.custom_initial_dataset_path:
            print("Loading custom initial dataset")
            self.custom_initial_dataset = load_w_pickle(self.custom_initial_dataset_path)
            if not os.path.exists(self.custom_initial_dataset_path + '.pkl'):
                raise ValueError(self.custom_initial_dataset_path + '.pkl')
            assert self.config['n_init'] == len(self.custom_initial_dataset), (
                self.config['n_init'], len(self.custom_initial_dataset))

        if self.config['kernel_type'] is None:
            self.config['kernel_type'] = 'transformed_overlap'
            print(f"Kernel Not Specified Using Default {self.config['kernel_type']}")

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        print(self.path)

        self.res = pd.DataFrame(np.nan, index=np.arange(int(self.config['max_iters'] * self.config['batch_size'])),
                                columns=['Index', 'LastValue', 'BestValue', 'Time', 'LastProtein', 'BestProtein'])

        self.nm_AAs = 20
        self.n_categories = np.array([self.nm_AAs] * self.config['seq_len'])
        self.start_itern = 0
        self.f_obj = CDRBO(self.config['device'], self.n_categories, self.config['seq_len'], self.config['bbox'], False)

    @staticmethod
    def get_path(save_path: str, antigen: str, kernel_type: str, seed: int, cdr_constraints: int, seq_len: int,
                 search_strategy: str,
                 custom_init_dataset_path: Optional[str] = None):
        path: str = f"{save_path}/BO_{kernel_type}/antigen_{antigen}" \
                    f"_kernel_{kernel_type}_search-strat_{search_strategy}_seed_{seed}" \
                    f"_cdr_constraint_{bool(cdr_constraints)}_seqlen_{seq_len}"
        if custom_init_dataset_path:
            custom_init_id = os.path.basename(os.path.dirname(custom_init_dataset_path))
            custom_init_id_seed = os.path.basename(os.path.dirname(os.path.dirname(custom_init_dataset_path)))
            path += f"_custom-init-id-{custom_init_id}_seed_{custom_init_id_seed}"
        return path

    @property
    def path(self) -> str:
        return self.get_path(
            save_path=self.config['save_path'],
            antigen=self.config['bbox']['antigen'],
            kernel_type=self.config['kernel_type'],
            search_strategy=self.config['search_strategy'],
            seed=self.seed,
            cdr_constraints=self.cdr_constraints,
            seq_len=self.config['seq_len'],
            custom_init_dataset_path=self.custom_initial_dataset_path
        )

    @property
    def custom_initial_dataset_path(self) -> Optional[str]:
        if not self.config.get('custom_init', False):
            return None
        return get_initial_dataset_path(
            antigen_name=self.config['bbox']['antigen'],
            n_per_cat=get_n_per_cat(n_loosers=self.config['custom_init_n_loosers'],
                                    n_mascottes=self.config['custom_init_n_mascottes'],
                                    n_heroes=self.config['custom_init_n_heroes']),
            top_cut_ratio_per_cat=get_top_cut_ratio_per_cat(
                top_cut_ratio_loosers=self.config['custom_init_top_cut_loosers'],
                top_cut_ratio_mascottes=self.config['custom_init_top_cut_mascottes'],
                top_cut_ratio_heroes=self.config['custom_init_top_cut_heroes']),
            seed=self.config['custom_init_seed']
        )

    @property
    def torch_rd_state_path(self) -> str:
        return os.path.join(self.path, 'torch_rd_state.pt')

    @property
    def np_rd_state_path(self) -> str:
        return os.path.join(self.path, "np_rd_state.pkl")

    @property
    def random_rd_state_path(self) -> str:
        return os.path.join(self.path, "random_rd_state.pkl")

    def load(self):
        res_path = os.path.join(self.path, 'results.csv')
        optim_path = os.path.join(self.path, 'optim.pkl')
        if os.path.exists(optim_path):
            optim = load_w_pickle(optim_path)
            if os.path.exists(self.torch_rd_state_path):
                torch_random_state = torch.load(self.torch_rd_state_path)
                torch.set_rng_state(torch_random_state)
            if os.path.exists(self.np_rd_state_path):
                np_rd_state = load_w_pickle(self.np_rd_state_path)
                np.random.set_state(np_rd_state)
            if os.path.exists(self.random_rd_state_path):
                rd_state = load_w_pickle(self.random_rd_state_path)
                random.setstate(rd_state)
        else:
            optim = None
        if os.path.exists(res_path):
            self.res = pd.read_csv(res_path,
                                   usecols=['Index', 'LastValue', 'BestValue', 'Time', 'LastProtein', 'BestProtein'])
            self.start_itern = len(self.res) - self.res['Index'].isna().sum() // self.config['batch_size']
        print(f"-- Resume -- Already observed {optim.casmopolitan.n_evals}")
        return optim

    def save(self, optim):
        optim_path = os.path.join(self.path, 'optim.pkl')
        res_path = os.path.join(self.path, 'results.csv')
        save_w_pickle(optim, optim_path)
        self.res.to_csv(res_path)
        # save random states
        torch.save(torch.get_rng_state(), self.torch_rd_state_path)
        save_w_pickle(np.random.get_state(), self.np_rd_state_path)
        save_w_pickle(random.getstate(), self.random_rd_state_path)

    def results(self, optim, x, itern, rtime):
        Y = np.array(optim.casmopolitan.fX)
        if Y[:(itern + 1)].shape[0]:

            # sequential
            if self.config['batch_size'] == 1:
                argmin = np.argmin(Y[:(itern + 1) * self.config['batch_size']])
                x_best = ''.join([self.f_obj.fbox.idx_to_AA[j] for j in
                                  optim.casmopolitan.X[:(itern + 1) * self.config['batch_size']][argmin].flatten()])
                self.res.iloc[itern, :] = [itern, float(Y[-1]), float(np.min(Y[:(itern + 1)])), rtime,
                                           self.f_obj.idx_to_seq(x)[0], x_best]
            # batch
            else:
                for idx, j in enumerate(
                        range(itern * self.config['batch_size'], (itern + 1) * self.config['batch_size'])):
                    argmin = np.argmin(Y[:(j + 1) * self.config['batch_size']])
                    x_best = ''.join([self.f_obj.fbox.idx_to_AA[ind] for ind in
                                      optim.casmopolitan.X[:(j + 1) * self.config['batch_size']][argmin].flatten()])
                    self.res.iloc[j, :] = [j, float(Y[-idx]), float(np.min(Y[:(j + 1) * self.config['batch_size']])),
                                           rtime,
                                           self.f_obj.idx_to_seq(x)[idx], x_best]

    def run(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        kwargs = {
            'length_max_discrete': self.config['seq_len'],
            'device': self.config['device'],
            'seed': self.seed,
            'search_strategy': self.search_strategy
        }

        if self.config['resume']:
            optim = self.load()
        else:
            optim = None

        if not optim:
            optim = Optimizer(
                self.n_categories, min_cuda=self.config['min_cuda'],
                n_init=self.config['n_init'], use_ard=self.config['ard'],
                acq=self.config['acq'],
                cdr_constraints=self.cdr_constraints,
                normalise=self.config['normalise'],
                kernel_type=self.config['kernel_type'],
                noise_variance=float(self.config['noise_variance']),
                alphabet_size=self.nm_AAs,
                **kwargs
            )

            if self.config.get("pre_evals") is not None:
                pre_eval_x, pre_eval_y = get_x_y_from_csv(self.config.get("pre_evals"))
                optim.suggest(len(pre_eval_x))  # exhaust init random suggestions
                optim.batch_size = self.config['batch_size']
                optim.casmopolitan.batch_size = optim.batch_size
                optim.casmopolitan.n_init = max([optim.casmopolitan.n_init, optim.batch_size])
                optim.observe(pre_eval_x, pre_eval_y)
                print(f"Observed {len(pre_eval_y)} already evaluated points")

        for itern in range(self.start_itern, self.config['max_iters']):
            start = time.time()
            x_next = optim.suggest(self.config['batch_size'])
            if self.custom_initial_dataset and len(optim.casmopolitan.fX) < self.config['n_init']:
                # observe the custom initial points instead of the suggested ones
                n_random = min(x_next.shape[0], self.config['n_init'] - len(optim.casmopolitan.fX))
                x_next[:n_random] = self.custom_initial_dataset.get_index_encoded_x()[
                                    len(optim.casmopolitan.fX):len(optim.casmopolitan.fX) + n_random]
            y_next = self.f_obj.compute(x_next)
            optim.observe(x_next, y_next)
            end = time.time()
            self.results(optim, x_next, itern, rtime=end - start)
            if itern % 5 == 0:
                self.log(f"Iter {itern + 1} / {self.config['max_iters']} in {end - start:.2f} s "
                         f"- {''.join(['ACDEFGHIKLMNPQRSTVWY'[int(x)] for x in x_next[0]])}"
                         f" ({y_next[0].item():.2f})")
            self.save(optim)

    def log(self, message: str, end: Optional[str] = None):
        log(message=message,
            header=f"BOExp - {self.config['bbox']['antigen']} - {self.config['kernel_type']} - seed {self.seed}",
            end=end)


from bo.utils import get_config

if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=True,
                                     description='Antigen-CDR3 binding prediction using high dimensional BO')
    parser.add_argument('--antigens_file', type=str, default='./dataloader/all_antigens.txt',
                        help='List of Antigen to perform BO')
    parser.add_argument('--seed', type=int, default=42, help='initial seed setting')
    parser.add_argument('--n_trials', type=int, default=3, help='number of random trials')
    parser.add_argument('--resume', type=bool, default=False, help='flag to resume training')
    parser.add_argument('--resume_trial', type=int, default=0, help='resume trial for training')
    parser.add_argument('--cdr_constraints', type=bool, default=True, help='constraint local search')
    parser.add_argument('--config', type=str, default='./bo/config.yaml',
                        help='Configuration File')
    args = parser.parse_args()
    config_ = get_config(os.path.abspath(args.config))
    config_['resume'] = args.resume

    with open(args.antigens_file) as file:
        antigens = file.readlines()
        antigens = [antigen.rstrip() for antigen in antigens]

    print(f'Iterating Over All Antigens In File {args.antigens_file} \n {antigens}')
    # antigens = ['1ADQ_A', '1FBI_X', '1HOD_C', '1NSN_S', '1OB1_C', '1WEJ_F', '2YPV_A', '3RAJ_A', '3VRL_C', '2DD8_S', '1S78_B', '2JEL_P']

    for antigen in antigens:
        start_antigen = time.time()
        seeds = list(range(args.seed, args.seed + args.n_trials))
        t = args.resume_trial
        while (t < args.n_trials):
            print(f"Starting Trial {t + 1} for antigen {antigen}")
            config_['bbox']['antigen'] = antigen

            boexp = BOExperiments(config_, args.cdr_constraints, seeds[t])

            try:
                boexp.run()
            except FileNotFoundError as e:
                print(e.args)
                continue

            del boexp
            torch.cuda.empty_cache()
            end_antigen = time.time()
            print(f"Time taken for antigen {antigen} trial {t + 1} = {end_antigen - start_antigen}")
            t += 1
        args.resume_trial = 0
    print('BO finished')

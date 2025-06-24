import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Any, Dict, get_args

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path.insert(0, ROOT_PROJECT)

from task import TableFilling
from task import BaseTool
from utilities.misc_utils import log
from bo.custom_init import get_initial_dataset_path, InitialBODataset, get_top_cut_ratio_per_cat, get_n_per_cat
from bo.botask import BOTask
from bo.localbo_utils import ACQ_FUNCTIONS, SEARCH_STRATS, AA
from bo.optimizer import Optimizer
from bo.utils import save_w_pickle, load_w_pickle


def get_x_y_from_csv(csv_path: str) -> tuple[np.ndarray, torch.tensor]:
    data = pd.read_csv(csv_path)
    x = data["x"].values
    from bo.localbo_utils import AA_to_idx
    x = np.array([[AA_to_idx[cat] for cat in xx] for xx in x])
    y = torch.tensor(data["y"].values).reshape(-1, 1)
    return x, y


class BOExperiments:
    def __init__(self, config: Dict[str, Any], cdr_constraints: bool, seed: int) -> None:
        """
        Args:
             config: dictionary of parameters for BO
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
                 vector_representation_table_csv: vector re
             seed: random seed
        """

        self.config = config
        self.table_of_antibodies_as_inds = None
        self.table_of_embeddings = None
        if self.config["tabular_search_csv"] is not None:
            print(f"Tab. BO setting: will select antibodies among available ones from: {config['tabular_search_csv']}")
            aux_tab = self.get_table_of_antibodies(tabular_search_csv=self.config["tabular_search_csv"])
            self.table_of_antibodies_as_inds, self.table_of_embeddings, self.embedding_from_array_dict = aux_tab
            if self.table_of_embeddings is not None:
                print("Will use pre-computing embeddings provided in the csv.")
        self.seed = seed
        self.cdr_constraints = cdr_constraints
        # Sanity checks
        if self.config['acq'] not in get_args(ACQ_FUNCTIONS):
            raise ValueError(f"Unknown acquisition function choice {self.config['acq']} (not in {ACQ_FUNCTIONS})")
        self.search_strat = self.config.get('search_strategy', 'local')
        assert self.search_strat in get_args(SEARCH_STRATS), print(f"{self.search_strat} not in {SEARCH_STRATS}")

        print(f"Search Strategy {self.search_strat}")

        self.custom_initial_dataset: Optional[InitialBODataset] = None
        if self.custom_initial_dataset_path:
            print("Loading custom initial dataset")
            self.custom_initial_dataset = load_w_pickle(self.custom_initial_dataset_path)
            if not os.path.exists(self.custom_initial_dataset_path + '.pkl'):
                raise ValueError(self.custom_initial_dataset_path + '.pkl')
            if self.config['n_init'] != len(self.custom_initial_dataset):
                raise ValueError(f"{self.config['n_init']} != {len(self.custom_initial_dataset)}")

        if self.config['kernel_type'] is None:
            default_kernel = 'transformed_overlap'
            self.config['kernel_type'] = default_kernel
            print(f"Kernel Not Specified Using Default {default_kernel}")

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        print(f"Results of this run will be saved in {self.path}")

        self.res = pd.DataFrame(np.nan, index=np.arange(int(self.config['max_iters'] * self.config['batch_size'])),
                                columns=['Index', 'LastValue', 'BestValue', 'Time', 'LastProtein', 'BestProtein'])

        self.nb_aas = len(AA)
        self.n_categories = np.array([self.nb_aas] * self.config['seq_len'])
        self.start_itern = 0
        self.f_obj = BOTask(
            device=self.config['device'], n_categories=self.n_categories,
            seq_len=self.config['seq_len'], bbox=self.config['bbox'], normalise=False
        )

    @staticmethod
    def get_path(save_path: str, antigen: str, kernel_type: str, seed: int, cdr_constraints: int, seq_len: int,
                 search_strategy: str,
                 custom_init_dataset_path: Optional[str] = None, tabular_search_csv: Optional[str] = None):
        path: str = f"{save_path}/BO_{kernel_type}/antigen_{antigen}" \
                    f"_kernel_{kernel_type}_search-strat_{search_strategy}_seed_{seed}" \
                    f"_cdr_constraint_{bool(cdr_constraints)}_seqlen_{seq_len}"
        if tabular_search_csv is not None:
            path += f"_tabsearch-{os.path.basename(tabular_search_csv)[:-4]}"
        if custom_init_dataset_path:
            custom_init_id = os.path.basename(os.path.dirname(custom_init_dataset_path))
            custom_init_id_seed = os.path.basename(os.path.dirname(os.path.dirname(custom_init_dataset_path)))
            path += f"_custom-init-id-{custom_init_id}_seed_{custom_init_id_seed}"
        return os.path.abspath(path)

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
            custom_init_dataset_path=self.custom_initial_dataset_path,
            tabular_search_csv=self.config["tabular_search_csv"]
        )

    @property
    def custom_initial_dataset_path(self) -> Optional[str]:
        if not self.config.get('custom_init', False):
            return None
        return get_initial_dataset_path(
            antigen_name=self.config['bbox']['antigen'],
            n_per_cat=get_n_per_cat(
                n_loosers=self.config['custom_init_n_loosers'],
                n_mascottes=self.config['custom_init_n_mascottes'],
                n_heroes=self.config['custom_init_n_heroes']
            ),
            top_cut_ratio_per_cat=get_top_cut_ratio_per_cat(
                top_cut_ratio_loosers=self.config['custom_init_top_cut_loosers'],
                top_cut_ratio_mascottes=self.config['custom_init_top_cut_mascottes'],
                top_cut_ratio_heroes=self.config['custom_init_top_cut_heroes']
            ),
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

    def load(self) -> Optimizer:
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
            if os.path.exists(res_path):
                columns = ['Index', 'LastValue', 'BestValue', 'Time', 'LastProtein', 'BestProtein']
                self.res = pd.read_csv(res_path, usecols=columns)
                self.start_itern = (len(self.res) - self.res['Index'].isna().sum()) // self.config['batch_size']
            print(f"-- Resume -- Already observed {optim.casmopolitan.n_evals}")
            return optim

    def save(self, optim: Optimizer) -> None:
        optim_path = os.path.join(self.path, 'optim.pkl')
        res_path = os.path.join(self.path, 'results.csv')
        save_w_pickle(obj=optim, path=optim_path)
        self.res.to_csv(res_path)
        # save random states
        torch.save(torch.get_rng_state(), self.torch_rd_state_path)
        save_w_pickle(obj=np.random.get_state(), path=self.np_rd_state_path)
        save_w_pickle(obj=random.getstate(), path=self.random_rd_state_path)

    def results(self, optim: Optimizer, x: np.ndarray, itern: int, rtime: float) -> None:
        y = np.array(optim.casmopolitan.fx)
        if y[:itern + 1].shape[0] == 0:
            return

        antibodies = self.f_obj.idx_to_seq(x)

        def add_res(step: int, y_val: float, protein: str) -> None:
            argmin = np.argmin(y[:step + 1])
            x_best = ''.join([self.f_obj.fbox.idx_to_AA[ind] for ind in optim.casmopolitan.x[argmin].flatten()])
            self.res.iloc[step, :] = [step, y_val, float(np.min(y[:(step + 1)])), rtime, protein, x_best]

        for idx, j in enumerate(range(itern * self.config['batch_size'], (itern + 1) * self.config['batch_size'])):
            add_res(step=j, y_val=float(y[j]), protein=antibodies[idx])

    def run(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        kwargs = {
            'length_max_discrete': self.config['seq_len'],
            'device': self.config['device'],
            'seed': self.seed,
            'search_strategy': self.search_strat,
            'BERT_model_path': self.config.get('BERT_model_path', 'Rostlab/prot_bert_bfd'),
            'BERT_tokeniser_path': self.config.get('BERT_tokenizer_path', 'Rostlab/prot_bert_bfd'),
            'BERT_batchsize': self.config.get('BERT_batchsize', 128),
        }

        if self.config['resume']:
            optim = self.load()
        else:
            optim = None

        if not optim:
            optim = Optimizer(
                config=self.n_categories, min_cuda=self.config['min_cuda'],
                n_init=self.config['n_init'], use_ard=self.config['ard'],
                acq=self.config['acq'],
                cdr_constraints=self.cdr_constraints,
                normalise=self.config['normalise'],
                kernel_type=self.config['kernel_type'],
                noise_variance=float(self.config['noise_variance']),
                alphabet_size=self.nb_aas,
                table_of_candidates=self.table_of_antibodies_as_inds,
                table_of_candidate_embeddings=self.table_of_embeddings,
                embedding_from_array_dict=self.embedding_from_array_dict,
                **kwargs
            )

            if self.config.get("pre_evals") is not None:
                pre_eval_x, pre_eval_y = get_x_y_from_csv(self.config.get("pre_evals"))
                optim.suggest(len(pre_eval_x))  # exhaust init random suggestions
                optim.batch_size = self.config['batch_size']
                optim.casmopolitan.batch_size = optim.batch_size
                optim.casmopolitan.n_init = max([optim.casmopolitan.n_init, optim.batch_size])
                optim.observe(x=pre_eval_x, y=pre_eval_y)
                print(f"Observed {len(pre_eval_y)} already evaluated points")

        # check if there are points that have been suggested and evaluated since the last antbo call
        if isinstance(self.f_obj.fbox, TableFilling) and os.path.exists(self.f_obj.fbox.path_to_eval_csv):
            table_of_results = pd.read_csv(self.f_obj.fbox.path_to_eval_csv, index_col=None)
            if np.all(table_of_results["Validate (0/1)"].values):
                print(f"Get already evaluated points from table {self.f_obj.fbox.path_to_eval_csv}")
                y = torch.tensor(table_of_results["Validate (0/1)"].values)
                x_seqs = table_of_results.Antibody.values
                # convert strings to array
                x_seqs_ind = np.array(
                    [np.array([self.f_obj.fbox.AA_to_idx[char] for char in x_seq]) for x_seq in x_seqs]
                )
                if optim.batch_size is None:
                    optim.batch_size = len(x_seqs)
                    optim.casmopolitan.batch_size = len(x_seqs)
                    optim.casmopolitan.n_init = max([optim.casmopolitan.n_init, optim.batch_size])
                    optim.restart()
                optim.observe(x=x_seqs_ind, y=y)
                self.results(optim=optim, x=x_seqs_ind, itern=self.start_itern, rtime=0)
                self.start_itern += 1
                self.save(optim=optim)
                self.f_obj.fbox.make_copy_eval_table()

        for itern in range(self.start_itern, self.config['max_iters']):
            start = time.time()
            x_next = optim.suggest(n_suggestions=self.config['batch_size'])
            if self.custom_initial_dataset and len(optim.casmopolitan.fx) < self.config['n_init']:
                # observe the custom initial points instead of the suggested ones
                n_random = min(x_next.shape[0], self.config['n_init'] - len(optim.casmopolitan.fx))
                x_next[:n_random] = self.custom_initial_dataset.get_index_encoded_x()[
                                    len(optim.casmopolitan.fx):len(optim.casmopolitan.fx) + n_random]
            y_next = self.f_obj.compute(x=x_next)
            optim.observe(x=x_next, y=y_next)
            end = time.time()
            self.results(optim=optim, x=x_next, itern=itern, rtime=end - start)
            if itern % 5 == 0:
                self.log(f"Iter {itern + 1} / {self.config['max_iters']} in {end - start:.2f} s "
                         f"- {''.join(['ACDEFGHIKLMNPQRSTVWY'[int(x)] for x in x_next[0]])}"
                         f" ({y_next[0].item():.2f})")
            self.save(optim=optim)

    def log(self, message: str, end: Optional[str] = None) -> None:
        header = f"BOExp - {self.config['bbox']['antigen']} - {self.config['kernel_type']} - seed {self.seed}"
        log(message=message, header=header, end=end)

    @staticmethod
    def get_table_of_antibodies(tabular_search_csv: str, normalize_embeddings: bool = True) \
            -> tuple[np.ndarray, Optional[np.ndarray], dict[str, np.ndarray]]:
        """ Return array of antigens where each row corresponds to an antibody given by the index of its AA

        Args:
            - tabular_search_csv: path to the csv file containing the AAs (and optionally vector representations)
            - normalize_embeddings: whether to min-max normalize the embeddings

        Returns:
            - aas_as_inds: array of aas (each entry is an array of AA indices)
            - embeddings: array of shape (n_antibodies, embedding size)
            - embedding_from_aas_as_inds_dict: dictionary mapping the antibody arrays to their embeddings
        """
        data = pd.read_csv(tabular_search_csv, index_col=None)
        if data.shape[-1] == 1:
            aas = data.values.flatten()
            embeddings = None
        else:
            assert np.all(data.columns[1:] == [f"d{i}" for i in range(1, data.shape[1])]), data.columns[1:]
            aas = data.values[:, 0]
            embeddings = data.values[:, 1:].astype(float)
            if normalize_embeddings:
                min_embeddings = embeddings.min(0)
                max_embeddings = embeddings.max(0)
                embeddings = (embeddings - min_embeddings) / (max_embeddings - min_embeddings)
        arr = np.array([list(c for c in x) for x in aas])
        aas_as_inds = BaseTool().convert_array_aas_to_idx(arr)
        if embeddings is not None:
            embedding_from_aas_as_inds_dict = {
                str(aas_as_inds[i].astype(int)): embeddings[i] for i in tqdm(range(len(aas_as_inds)))
            }
        else:
            embedding_from_aas_as_inds_dict = None
        return aas_as_inds, embeddings, embedding_from_aas_as_inds_dict


if __name__ == '__main__':
    from bo.utils import get_config

    parser = argparse.ArgumentParser(add_help=True, description='Antigen-CDR3 binding prediction using BO')
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
    # antigens = ['1ADQ_A', '1FBI_X', '1HOD_C', '1NSN_S', '1OB1_C', '1WEJ_F',
    # '2YPV_A', '3RAJ_A', '3VRL_C', '2DD8_S', '1S78_B', '2JEL_P']

    for antigen in antigens:
        start_antigen = time.time()
        seeds = list(range(args.seed, args.seed + args.n_trials))
        t = args.resume_trial
        while t < args.n_trials:
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

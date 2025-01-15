import os
import pickle
from typing import Any, Optional

import numpy as np


def spearman(pred, target) -> float:
    """Compute the spearman correlation coefficient between prediction and target"""
    from scipy import stats
    coef_val, p_val = stats.spearmanr(pred, target)
    return coef_val


def pearson(pred, target) -> float:
    from scipy import stats
    coef_val, p_val = stats.pearsonr(pred, target)
    return coef_val


def negative_log_likelihood(pred, pred_std, target) -> float:
    """Compute the negative log-likelihood on the validation dataset"""
    from scipy.stats import norm
    n = pred.shape[0]
    res = 0.
    for i in range(n):
        res += (
            np.log(norm.pdf(target[i], pred[i], pred_std[i])).sum()
        )
    return -res


def get_dim_info(n_categories):
    dim_info = []
    offset = 0
    for i, cat in enumerate(n_categories):
        dim_info.append(list(range(offset, offset + cat)))
        offset += cat
    return dim_info


def save_w_pickle(obj: Any, path: str, filename: Optional[str] = None) -> None:
    """ Save object obj in file exp_path/filename.pkl """
    if filename is None:
        filename = os.path.basename(path)
        path = os.path.dirname(path)
    if len(filename) < 4 or filename[-4:] != '.pkl':
        filename += '.pkl'
    with open(os.path.join(path, filename), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_w_pickle(path: str, filename: Optional[str] = None) -> Any:
    """ Load object from file exp_path/filename.pkl """
    if filename is None:
        filename = os.path.basename(path)
        path = os.path.dirname(path)
    if len(filename) < 4 or filename[-4:] != '.pkl':
        filename += '.pkl'
    with open(os.path.join(path, filename), 'rb') as f:
        try:
            return pickle.load(f)
        except EOFError as e:
            print(path, filename)
            raise


import yaml


def get_config(config):
    with open(config, 'r') as f:
        return yaml.safe_load(f)


def batch_iterator(data1, step=8):
    size = len(data1)
    for i in range(0, size, step):
        yield data1[i:min(i + step, size)]


import torch


class BERTFeatures:
    """Compute BERT Features"""

    def __init__(self, model, tokeniser):
        AAs = 'ACDEFGHIKLMNPQRSTVWY'
        self.AA_to_idx = {aa: i for i, aa in enumerate(AAs)}
        self.idx_to_AA = {value: key for key, value in self.AA_to_idx.items()}
        self.model = model
        self.tokeniser = tokeniser

    def compute_features(self, x1):
        assert x1.ndim == 2
        inp_device = x1.device
        self.model = self.model.to(inp_device)
        with torch.no_grad():
            x1 = [" ".join(self.idx_to_AA[i.item()] for i in x_i) for x_i in x1]
            ids1 = self.tokeniser.batch_encode_plus(x1, add_special_tokens=False, padding=True)
            input_ids1 = torch.tensor(ids1['input_ids']).to(inp_device)
            attention_mask1 = torch.tensor(ids1['attention_mask']).to(inp_device)
            reprsn1 = self.model.to(inp_device)(input_ids=input_ids1, attention_mask=attention_mask1)[0]
        return reprsn1.mean(1)


def update_table_of_candidates(
        original_table: np.ndarray,
        observed_candidates: np.ndarray, check_candidates_in_table: bool,
        table_of_candidate_embeddings: Optional[np.ndarray]
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """ Update the table of candidates, removing the newly observed candidates from the table

    Args:
        original_table: table of candidates before observation
        observed_candidates: new observed points
        check_candidates_in_table: whether the observed candidates should be in the original_table or not
        table_of_candidate_embeddings: if not None, the embeddings of the candidates should be used to build the
            surrogate model
    Returns:
          Updated original_table and embeddings
    """
    if observed_candidates.ndim == 1:
        observed_candidates = observed_candidates.reshape(1, -1)
    for candidate in observed_candidates:
        filtr = np.all(original_table == candidate.reshape(1, -1), axis=1)
        if not np.any(filtr) and check_candidates_in_table:
            raise RuntimeError(f"New point {candidate} is not in the table of candidates.")
        original_table = original_table[~filtr]
        if table_of_candidate_embeddings is not None:
            table_of_candidate_embeddings = table_of_candidate_embeddings[~filtr]
    return original_table, table_of_candidate_embeddings


def update_table_of_candidates_array(original_table: np.ndarray, observed_candidates: np.ndarray,
                                     check_candidates_in_table: bool) -> np.ndarray:
    """ Update the table of candidates, removing the newly observed candidates from the table

    Args:
        original_table: table of candidates before observation
        observed_candidates: new observed points
        check_candidates_in_table: whether the observed candidates should be in the original_table or not

    Returns:
          Updated table
    """
    if observed_candidates.ndim == 1:
        observed_candidates = observed_candidates.reshape(1, -1)
    for candidate in observed_candidates:
        filtr = np.all(original_table == candidate.reshape(1, -1), axis=1)
        if not np.any(filtr) and check_candidates_in_table:
            raise RuntimeError(f"New point {candidate} is not in the table of candidates.")
        original_table = original_table[~filtr]
    return original_table


if __name__ == '__main__':
    bert_config = {'datapath': '/nfs/aiml/asif/CDRdata',
                   'path': '/nfs/aiml/asif/ProtBERT',
                   'modelname': 'OutputFinetuneBERTprot_bert_bfd',
                   'use_cuda': True,
                   'batch_size': 256
                   }
    device_ids = [2, 3]
    import glob
    import numpy as np

    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(id_) for id_ in device_ids)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer, \
        AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() and bert_config['use_cuda'] else "cpu")
    tokeniser = AutoTokenizer.from_pretrained(f"{bert_config['path']}/{bert_config['modelname']}")
    model = AutoModel.from_pretrained(f"{bert_config['path']}/{bert_config['modelname']}").to(device)
    bert_features = BERTFeatures(model, tokeniser)

    antigens = ['1ADQ_A', '1FBI_X', '1H0D_C', '1NSN_S', '1OB1_C', '1WEJ_F', '2YPV_A', '3RAJ_A', '3VRL_C', '2DD8_S',
                '1S78_B', '2JEL_P']

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from joblib import dump
    import pandas as pd

    for antigen in antigens:
        print(f"PCA for antigen {antigen}")
        try:
            filenames = glob.glob(f"{bert_config['datapath']}/RawBindingsMurine/{antigen}/*.txt")
            for i in range(len(filenames)):
                try:
                    if i == 0:
                        sequences = pd.read_csv(filenames[i], skiprows=1, sep='\t')
                        sequences = list(set(sequences['Slide'].dropna().values))
                    else:
                        df_i = pd.read_csv(filenames[i], skiprows=1, sep='\t')
                        df_i = list(set(df_i['Slide'].dropna().values))
                        sequences = list(set(sequences + df_i))
                except pd.errors.ParserError as err:
                    print(f"{filenames[i]} causes an error {err}")
                    continue
        except:
            continue

        if len(filenames) != 0:
            reprsns = []
            for seq_batch in batch_iterator(sequences, bert_config['batch_size']):
                seq_batch = torch.tensor([[bert_features.AA_to_idx[aa] for aa in seq] for seq in seq_batch]).to(device)
                seq_reprsn = bert_features.compute_features(seq_batch)
                reprsns.append(seq_reprsn.cpu().numpy())
                if len(reprsns) == 1000:
                    break
            reprsns = np.concatenate(reprsns, 0)
            scaler = StandardScaler()
            scaler.fit(reprsns)
            scaled_reprsns = scaler.transform(reprsns)
            pca = PCA(n_components=100)
            pca.fit(scaled_reprsns)
            results_path = f"{bert_config['datapath']}/finetune_pca"
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            dump(pca, f"{results_path}/{antigen}_pca.joblib")
            dump(scaler, f"{results_path}/{antigen}_scaler.joblib")

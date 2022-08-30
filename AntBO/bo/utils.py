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
    import numpy as np
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

from typing import Any, List, Optional
import os
import pickle


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


import os
from einops import rearrange

def batch_iterator(data1, step=8):
    size = len(data1)
    for i in range(0, size, step):
        yield data1[i:min(i+step, size)]

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
        inp_device = x1.device
        with torch.no_grad():
            x1 = [" ".join(self.idx_to_AA[i.item()] for i in x_i) for x_i in x1]
            ids1 = self.tokeniser.batch_encode_plus(x1, add_special_tokens=False, padding=True)
            input_ids1 = torch.tensor(ids1['input_ids']).to(self.model.device)
            attention_mask1 = torch.tensor(ids1['attention_mask']).to(self.model.device)
            reprsn1 = self.model(input_ids=input_ids1, attention_mask=attention_mask1)[0]
        return reprsn1.to(inp_device)

if __name__=='__main__':
    bert_config = { 'datapath': '/nfs/aiml/asif/CDRdata',
                    'path': '/nfs/aiml/asif/ProtBERT',
                   'modelname': 'prot_bert_bfd',
                    'use_cuda': True,
                    'batch_size': 256
                   }
    device_ids = [2,3]
    import os
    import glob
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(id) for id in device_ids)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import pipeline, \
        AutoTokenizer, \
        Trainer, \
        AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() and bert_config['use_cuda'] else "cpu")
    tokeniser = AutoTokenizer.from_pretrained(f"{bert_config['path']}/{bert_config['modelname']}")
    model = AutoModel.from_pretrained(f"{bert_config['path']}/{bert_config['modelname']}").to(device)
    bert_features = BERTFeatures(model, tokeniser)

    #antigens = ['1ADQ_A', '1FBI_X', '1HOD_C', '1NSN_S', '1OB1_C', '1WEJ_F', '2YPV_A', '3RAJ_A', '3VRL_C']
    antigens = [antigen.strip().split()[1] for antigen in open(f"/nfs/aiml/asif/CDRdata/antigens.txt", 'r') if antigen != '\n']

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from joblib import dump, load
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
            reprsns = []
            for seq_batch in batch_iterator(sequences, bert_config['batch_size']):
                seq_batch = torch.tensor([[bert_features.AA_to_idx[aa] for aa in seq] for seq in seq_batch]).to(device)
                seq_reprsn = bert_features.compute_features(seq_batch)
                seq_reprsn = rearrange(seq_reprsn, 'b l d -> b (l d)')
                reprsns.append(seq_reprsn)
                if len(reprsns) == 1000:
                    break

            reprsns = torch.cat(reprsns, 0).cpu().numpy()
            scaler = StandardScaler()
            scaler.fit(scaled_reprsns)
            scaled_reprsns = scaler.transform(reprsns)
            pca = PCA(n_components=100)
            pca.fit(scaled_reprsns)
            dump(pca, f"{bert_config['datapath']}/{antigen}_pca.joblib")
            dump(scaler, f"{bert_config['datapath']}/{antigen}_scaler.joblib")
        except:
            continue

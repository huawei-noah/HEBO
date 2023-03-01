import botorch.optim
import pandas as pd
import torch
import os, sys
import tqdm
import numpy as np

from datetime import datetime
from pathlib import Path
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from torch.utils.data import DataLoader

ROOT = str(Path(os.path.realpath(__file__)).parent.parent.parent)
SAVED_MODELS = os.path.join(ROOT, 'saved_models')
BO_RESULTS = os.path.join(ROOT, 'lsbo/results')
sys.path.insert(0, ROOT)

from lsbo.utils.load_utils import load_transvae
from lsbo.models.cnn import CNNRegModel, AbDataset
from lsbo.utils.bo_utils import evaluate_seq_absolut
from TransVAE.transvae.tvae_util import decode_mols

# load TransVAE
# load CNN
# create GP model
# sample initial points in latent space z1,...,zN and their bbox value v1,...,vN
# at each iteration:
#   fit GP on D=(Z,V)
#   optimise acqf. wrt. z and suggest next z_new
#   evaluate z_new in bbox:
#       decode z_new to a distribution of sequences (s1,...,sK)
#       evaluate all sequences in Absolut! to get their binding energy Absolut!(s)=e
#       use this dataset T=(S,E) to train the CNN from a fixed checkpoint
#       use the CNN to classify the real dataset points and get classification accuracy v_new
#   add (z_new, v_new) to D
#   repeat until end of budget

if __name__ == '__main__':
    LATENT_DIM = 2
    N_LATENT_POINTS = 3
    BO_STEPS = 1000
    THRESHOLD = -116.81 # top-0.35% of HER2Absolut! dataset energies
    BATCH_SIZE = 1024
    PRE_TRAIN_EPOCHS = 100
    FINE_TUNE_EPOCHS = 100
    X_LOWER_BOUND = [-5, -12] * N_LATENT_POINTS
    X_UPPER_BOUND = [6, 10] * N_LATENT_POINTS
    N_SAMPLES = 500
    N_FOLDS = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = os.path.join(BO_RESULTS, datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cnn_path = os.path.join(SAVED_MODELS, 'cnn/cnn_init.pt')

    ### Load trained TransVAE
    tvae_path = os.path.join(SAVED_MODELS, 'transvae32-2-her2-prop_pred-logratio/best.ckpt')
    tvae = load_transvae(tvae_path)
    tvae.model.eval()

    ### Save configuration
    with open(os.path.join(save_dir, 'conf.txt'), 'w') as conf:
        conf.write(f'LATENT_DIM: {LATENT_DIM}\n')
        conf.write(f'N_LATENT_POINTS: {N_LATENT_POINTS}\n')
        conf.write(f'BO_STEPS: {BO_STEPS}\n')
        conf.write(f'THRESHOLD: {THRESHOLD}\n')
        conf.write(f'BATCH_SIZE: {BATCH_SIZE}\n')
        conf.write(f'PRE_TRAIN_EPOCHS: {PRE_TRAIN_EPOCHS}\n')
        conf.write(f'FINE_TUNE_EPOCHS: {FINE_TUNE_EPOCHS}\n')
        conf.write(f'X_LOWER_BOUND: {X_LOWER_BOUND}\n')
        conf.write(f'X_UPPER_BOUND: {X_UPPER_BOUND}\n')
        conf.write(f'N_SAMPLES: {N_SAMPLES}\n')
        conf.write(f'N_FOLDS: {N_FOLDS}\n')
        conf.write(f'DEVICE: {DEVICE}\n')
        conf.write(f'TVAE: {os.path.join(SAVED_MODELS, "transvae32-2-her2-prop_pred-logratio/best.ckpt")}\n')
        conf.write(f'CNN-INIT: {os.path.join(SAVED_MODELS, "cnn/cnn_init.pt")}\n')

    ### Create a GP model in latent space
    x_init = None
    y_init = None
    xbounds = torch.stack((torch.tensor(X_LOWER_BOUND), torch.tensor(X_UPPER_BOUND))).to(torch.float)
    x = x_init
    y = y_init

    gp = None
    mll = None

    bo_pbar = tqdm.tqdm(range(BO_STEPS))
    for it in bo_pbar:
        ### Load trained CNN
        cnn = CNNRegModel(learning_rate=1e-5)
        cnn.load_state_dict(torch.load(cnn_path))

        ### fit GP model
        if gp is not None and mll is not None:
            _ = fit_gpytorch_mll(mll)

            alpha = UpperConfidenceBound(model=gp, beta=2.0, maximize=True)

            ### Optimise acquisition function
            candidate, _ = botorch.optim.optimize_acqf(
                acq_function=alpha,
                bounds=xbounds,
                num_restarts=100,
                raw_samples=500,
                q=1,
                return_best_only=True,
            )
        else:
            candidate = torch.randn((1, LATENT_DIM * N_LATENT_POINTS))

        #######################################################################
        ### Query candidate in black-box
        #######################################################################
        ### decode z_new to a distribution of sequences (s1,...,sK)
        ### evaluate all sequences in Absolut! to get their binding energy Absolut!(s)=e
        ### and save this new dataset
        candidates = candidate.view(N_LATENT_POINTS, LATENT_DIM)
        sequences = []
        for cand in candidates: # generate N_SAMPLES sequences per latent candidate
            cand_decoded = tvae.stochastic_decode(
                cand.view(1, -1).to(DEVICE),
                n_samples=N_SAMPLES,
                src_mask=torch.ones((1, 13)).to(bool).to(DEVICE)
            )
            sequences += decode_mols(cand_decoded, tvae.params['ORG_DICT'], fast=True)
        sequences_full = ['R' + seq[:-1] + 'Y' for seq in sequences]
        sequences_10 = [seq[:-1] for seq in sequences]
        energies = evaluate_seq_absolut(sequences_full, pdb_id='1S78_B',
                                        AbsolutNoLib_dir=ROOT,
                                        save_dir=os.path.join(ROOT, 'lsbo/'),
                                        first_cpu=0, num_cpus=49)
        labels = np.where(energies > THRESHOLD, 0, 1)
        cnn_data = np.concatenate((np.array(sequences_10).reshape(-1, 1), labels, energies,
                                   (energies - energies.mean()) / energies.std()), axis=-1)
        cnn_data_df = pd.DataFrame(cnn_data, columns=['AASeq', 'AgClass', 'Energy', 'EnergyStd'])
        cnn_data_df.to_csv(os.path.join(ROOT, 'lsbo/bo_data/cnn_data/trainset.tsv'), sep='\t', index=False)

        ### train the CNN on generated sequences from VAE
        pre_train_dataset = AbDataset(path=os.path.join(ROOT, 'lsbo/bo_data/cnn_data/trainset.tsv'), objective_name='EnergyStd')
        pre_train_dataloader = DataLoader(pre_train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        pre_train_opt = cnn.configure_optimizers()
        cnn.train()
        cnn.regression()
        cnn.to(DEVICE)

        pre_train_pbar = tqdm.tqdm(range(PRE_TRAIN_EPOCHS))
        for _ in pre_train_pbar:
            epoch_avg_pre_train_loss = []
            for _, pre_train_batch in enumerate(pre_train_dataloader):
                pre_train_seq, pre_train_lab = pre_train_batch
                pre_train_seq, pre_train_lab = pre_train_seq.to(DEVICE), pre_train_lab.to(DEVICE)
                pre_train_pred = cnn(pre_train_seq)
                pre_train_opt.zero_grad()
                pre_train_loss = cnn.loss_fn(pre_train_pred.view(-1), pre_train_lab.to(pre_train_seq))
                pre_train_loss.backward()
                pre_train_opt.step()
                epoch_avg_pre_train_loss.append(pre_train_loss.item())
            epoch_avg_pre_train_loss = np.mean(epoch_avg_pre_train_loss)
            pre_train_pbar.set_description(f'Pre-Training Loss={epoch_avg_pre_train_loss.item():.5f}')

        torch.save(cnn.state_dict(), os.path.join(ROOT, 'lsbo/bo_data/cnn_data/pre_trained_cnn.pt'))

        ### Fine-tune and test on real data (Mason), N_FOLDS times
        mason_dataset = pd.read_csv(os.path.join(ROOT, 'data/mason_cleaned.tsv'), sep='\t')
        chunk_size = len(mason_dataset) // N_FOLDS
        test_accuracies = []

        k_folds_pbar = tqdm.tqdm(range(N_FOLDS))
        for k in k_folds_pbar:
            del cnn
            cnn = CNNRegModel(learning_rate=1e-5)
            cnn.load_state_dict(torch.load(os.path.join(ROOT, 'lsbo/bo_data/cnn_data/pre_trained_cnn.pt')))
            cnn.classification()
            ft_opt = cnn.configure_optimizers()
            cnn.to(DEVICE)
            cnn.train()

            test_idx = np.arange(len(mason_dataset))[k * chunk_size:(k + 1) * chunk_size]
            ft_idx = [i for i in range(len(mason_dataset)) if i not in test_idx]
            ft_dataset = AbDataset(mode='train', df=mason_dataset, indexes=ft_idx)
            ft_dataloader = DataLoader(ft_dataset, batch_size=BATCH_SIZE, shuffle=True)

            for _ in range(FINE_TUNE_EPOCHS):
                for _, batch in enumerate(ft_dataloader):
                    seq, lab = batch
                    seq, lab = seq.to(DEVICE), lab.to(DEVICE)
                    pred = cnn(seq)
                    ft_opt.zero_grad()
                    ft_loss = cnn.loss_fn(pred.view(-1), lab.to(seq))
                    ft_loss.backward()
                    ft_opt.step()

            ### use the CNN to classify the real dataset points and get classification accuracy v_new
            cnn.eval()
            test_dataset = AbDataset(mode='test', df=mason_dataset, indexes=test_idx)
            test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            avg_test_acc = 0.
            for _, batch in enumerate(test_dataloader):
                seq, lab = batch
                seq, lab = seq.to(DEVICE), lab.to(DEVICE)
                pred = cnn(seq)
                avg_test_acc += cnn.classification_accuracy(pred.view(-1), lab.to(seq))
            avg_test_acc /= len(test_dataset)
            test_accuracies.append(avg_test_acc.item())

            k_folds_pbar.set_description(f'[{k + 1}/{N_FOLDS}] Test Accuracy: '
                                         f'mean={torch.tensor(test_accuracies).mean().item():.3f} '
                                         f'std={torch.tensor(test_accuracies).std().item():.3f}')

        objective_value = torch.tensor(test_accuracies).mean().view(1)
        #######################################################################
        #######################################################################

        ### Update GP
        if x is None and y is None:
            x = candidate.view(1, -1)
            y = objective_value
        else:
            x = torch.cat((x, candidate), dim=0)
            y = torch.cat((y, objective_value))
        del gp
        x = x.to(float)
        y = y.to(float)
        gp = SingleTaskGP(
            train_X=x,
            train_Y=y.view(-1, 1),
            input_transform=Normalize(d=x.shape[-1], bounds=xbounds),
            outcome_transform=Standardize(m=1)
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

        ### Save models and data
        torch.save(gp.state_dict(), os.path.join(save_dir, 'gp.pt'))
        torch.save(x, os.path.join(save_dir, 'x.pt'))
        torch.save(y, os.path.join(save_dir, 'y.pt'))
        cnn_data_df.to_csv(os.path.join(save_dir, f'energy-iter{it}.tsv'), sep='\t', index=False)

        bo_pbar.set_description(f'New y={objective_value.item():.4f} | Best y={y.max().item()}')

        if objective_value.item() == y.max().item():
            print('Saving new best CNN model')
            torch.save(cnn.state_dict(), os.path.join(save_dir, f'cnn-iter{it}.pt'))

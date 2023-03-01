import os, sys
import pandas as pd
import numpy as np
from pathlib import Path

import torch
import tqdm

from lsbo.models.cnn import CNNRegModel, AbDataset
from torch.utils.data import DataLoader, Dataset

ROOT = str(Path(os.path.realpath(__file__)).parent.parent.parent)
EPOCHS = 100
BATCH_SIZE = 1024
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_FOLDS = 10

# mason = pd.read_csv(os.path.join(ROOT, 'data/mason_cleaned.tsv'), sep='\t')
dataset = AbDataset(path=os.path.join(ROOT, 'data/mason_cleaned.tsv'))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# cnn = CNNRegModel(learning_rate=1e-5)
# torch.save(cnn.state_dict(), os.path.join(ROOT, 'saved_models/cnn/cnn_init.pt'))

mason_dataset = pd.read_csv(os.path.join(ROOT, 'data/mason_cleaned.tsv'), sep='\t')
chunk_size = len(mason_dataset) // N_FOLDS
test_accuracies = []

for k in range(N_FOLDS):
    cnn = CNNRegModel(learning_rate=1e-5)
    cnn.load_state_dict(torch.load(os.path.join(ROOT, 'saved_models/cnn/cnn_init.pt')))
    cnn.classification()
    train_opt = cnn.configure_optimizers()
    cnn.to(DEVICE)
    cnn.train()

    test_idx = np.arange(len(mason_dataset))[k * chunk_size:(k + 1) * chunk_size]
    train_idx = [i for i in range(len(mason_dataset)) if i not in test_idx]
    train_dataset = AbDataset(mode='train', df=mason_dataset, indexes=train_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_pbar = tqdm.tqdm(range(EPOCHS))
    for epoch in train_pbar:
        train_epoch_avg_loss = []
        for _, batch in enumerate(train_dataloader):
            seq, lab = batch
            seq, lab = seq.to(DEVICE), lab.to(DEVICE)
            pred = cnn(seq)
            train_opt.zero_grad()
            train_loss = cnn.loss_fn(pred.view(-1), lab.to(seq))
            train_loss.backward()
            train_opt.step()
            train_epoch_avg_loss.append(train_loss.item())
        train_epoch_avg_loss = np.mean(train_epoch_avg_loss)
        train_pbar.set_description(f'loss={train_epoch_avg_loss.item():.5f}')

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
    print(f'[{k+1}/{N_FOLDS}] Test Accuracy: {avg_test_acc.item():.5f}')
    test_accuracies.append(avg_test_acc.item())
    del cnn

test_accuracies = np.array(test_accuracies)
print(f'Test Accuracy on {N_FOLDS} folds: mean={test_accuracies.mean():.5f} std={test_accuracies.std():.5f}')
np.save(os.path.join(ROOT, f'saved_models/cnn/{N_FOLDS}_folds_mean_test_accuracy.npy'), test_accuracies.mean())
np.save(os.path.join(ROOT, f'saved_models/cnn/{N_FOLDS}_folds_std_test_accuracy.npy'), test_accuracies.std())

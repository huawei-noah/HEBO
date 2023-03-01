from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import KFold

from torch import optim, nn, utils, Tensor
from sklearn.metrics import roc_curve, auc, precision_recall_curve, \
    average_precision_score, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.metrics import log_loss


def one_hot_encoder(s):
    alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    d = {a: i for i, a in enumerate(alphabet)}
    x = np.zeros((len(d), len(s)))
    x[[d[c] for c in s], range(len(s))] = 1
    return x


class AbDataset(Dataset):
    def __init__(self, path=None, r=1.0, mode='load', indexes=None, df=None, objective_name=None, random_seed=42):

        if mode == 'load':
            self.path = path
            self.df = pd.read_csv(self.path, sep='\t')
            np.random.seed(random_seed)
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            if r < 1.0:
                self.df = self.df[:int(self.df.shape[0] * r)]
            self.x = torch.tensor(
                np.array([one_hot_encoder(xi) for xi in self.df.AASeq]).reshape(-1, 1, 10, 20)).float()
            if objective_name is None:
                self.y = torch.tensor(self.df[['AgClass']].to_numpy().reshape(-1)).long()
            else:
                self.y = torch.tensor(self.df[[objective_name]].to_numpy().reshape(-1)).long()

        elif mode == 'train':
            self.df = df.loc[indexes]
            self.x = torch.tensor(
                np.array([one_hot_encoder(xi) for xi in self.df.AASeq]).reshape(-1, 1, 10, 20)).float()
            if objective_name is None:
                self.y = torch.tensor(self.df[['AgClass']].to_numpy().reshape(-1)).long()
            else:
                self.y = torch.tensor(self.df[[objective_name]].to_numpy().reshape(-1))

        else:
            self.df = df.loc[indexes]
            self.x = torch.tensor(
                np.array([one_hot_encoder(xi) for xi in self.df.AASeq]).reshape(-1, 1, 10, 20)).float()
            if objective_name is None:
                self.y = torch.tensor(self.df[['AgClass']].to_numpy().reshape(-1)).long()
            else:
                self.y = torch.tensor(self.df[[objective_name]].to_numpy().reshape(-1))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# class AbLightDataset(pl.LightningDataModule):
#     def __init__(
#             self,
#             random_seed,
#             path='../Data/',
#             k: int = 1,
#             batch_size: int = 16,
#             num_splits: int = 5,
#             r: float = 1.0,
#             data_f='data.tsv'):
#         super().__init__()
#         self.save_hyperparameters(logger=False)
#         self.batch_size = batch_size
#         self.path = path
#         self.random_seed = random_seed
#         self.r = r
#         self.data_f = data_f
#
#     def setup(self, stage=None):
#         self.data = AbDataset(path=self.path, r=self.r, random_seed=self.random_seed, data_f=self.data_f)
#
#         kf = KFold(n_splits=self.hparams.num_splits, shuffle=True, random_state=self.hparams.random_seed)
#
#         all_splits = [k for k in kf.split(self.data.df)]
#
#         train_indexes, val_indexes = all_splits[self.hparams.k]
#         train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()
#
#         self.data_train = AbDataset(mode='train', indexes=train_indexes, df=self.data.df)
#         self.data_val = AbDataset(mode='val', indexes=val_indexes, df=self.data.df)
#
#     def train_dataloader(self):
#         train_loader = DataLoader(self.data_train,
#                                   batch_size=self.batch_size,
#                                   shuffle=True)
#         return train_loader
#
#     def val_dataloader(self):
#         valid_loader = DataLoader(self.data_val,
#                                   batch_size=self.batch_size,
#                                   shuffle=False)
#         return valid_loader
#
#     def predict_dataloader(self):
#         valid_loader = DataLoader(self.data_val,
#                                   batch_size=self.batch_size,
#                                   shuffle=False)
#         return valid_loader


class CNNModel(nn.Module):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.conv = nn.Conv2d(in_channels=1, out_channels=400, kernel_size=(5, 20), padding=(2, 0))
        self.drop = nn.Dropout(p=0.2)
        self.pool = nn.MaxPool2d((2, 1), stride=1)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(3600, 300)
        self.fc2 = nn.Linear(300, 1)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.story_val = []

    def forward(self, x):
        x = self.conv(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return x

    def loss_fn(self, y_hat, y):
        return nn.BCELoss()(y_hat, y)

    def classification_accuracy(self, y_hat, y_true):
        pred_label = (y_hat > 0.5).to(int)
        return (pred_label == y_true).sum()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat.flatten(), y.float())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat.flatten(), y.float())
        fpr, tpr, thresholds = roc_curve(y, y_hat)
        roc_auc = auc(fpr, tpr)
        self.log('valid_loss', loss)
        self.log('roc auc', roc_auc)
        return loss, roc_auc

    def validation_epoch_end(self, outputs) -> None:
        loss = (sum(output[0] for output in outputs) / len(outputs)).item()
        roc_auc = np.nansum([output[1] for output in outputs]) / len(outputs)
        self.story_val.append([self.current_epoch, loss, roc_auc])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), eps=1e-07, lr=self.learning_rate)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat.flatten(), y.float())
        fpr, tpr, thresholds = roc_curve(y, y_hat)
        roc_auc = auc(fpr, tpr)
        return y_hat, y


class CNNRegModel(nn.Module):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.conv = nn.Conv2d(in_channels=1, out_channels=400, kernel_size=(5, 20), padding=(2, 0))
        self.drop = nn.Dropout(p=0.2)
        self.pool = nn.MaxPool2d((2, 1), stride=1)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(3600, 300)
        self.fc2 = nn.Linear(300, 1)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.lin = nn.Linear(1, 1)
        self.story_val = []
        self._regression_mode: bool = False

    def forward(self, x):
        x = self.conv(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.act1(self.fc1(x))
        x = self.fc2(x)
        if self._regression_mode:
            return x
        else:
            return self.act2(self.lin(x))

    def loss_fn(self, y_hat, y):
        if self._regression_mode:
            return nn.MSELoss()(y_hat, y)
        else:
            return nn.BCELoss()(y_hat, y)

    def regression(self):
        self._regression_mode = True

    def classification(self):
        self._regression_mode = False

    def classification_accuracy(self, y_hat, y_true):
        pred_label = (y_hat > 0.5).to(int)
        return (pred_label == y_true).sum()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat.flatten(), y.float())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat.flatten(), y.float())
        fpr, tpr, thresholds = roc_curve(y, y_hat)
        roc_auc = auc(fpr, tpr)
        self.log('valid_loss', loss)
        self.log('roc auc', roc_auc)
        return loss, roc_auc

    def validation_epoch_end(self, outputs) -> None:
        loss = (sum(output[0] for output in outputs) / len(outputs)).item()
        roc_auc = np.nansum([output[1] for output in outputs]) / len(outputs)
        self.story_val.append([self.current_epoch, loss, roc_auc])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), eps=1e-07, lr=self.learning_rate)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat.flatten(), y.float())
        fpr, tpr, thresholds = roc_curve(y, y_hat)
        roc_auc = auc(fpr, tpr)
        return y_hat, y


def f1(mod, dx, trainer):
    predictions = trainer.predict(mod, dx)
    preds = [p[0] for p in predictions]
    preds = [float(item) for sublist in preds for item in sublist]
    true = [p[1] for p in predictions]
    true = [float(item) for sublist in true for item in sublist]
    fpr, tpr, thresholds = roc_curve(true, preds)
    roc_auc = auc(fpr, tpr)
    lloss = log_loss(true, preds)
    return roc_auc, lloss
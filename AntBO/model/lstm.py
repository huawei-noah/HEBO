import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import Module

from dataloader.base import MultiModalDataLoader
from model.base import BaseModel


class LSTMModel(BaseModel, Module):
    def __init__(self, config):
        BaseModel.__init__(self)
        Module.__init__(self)
        self.config = config
        self.create_network()

    def create_network(self):
        args = self.configuration["model"]
        self.vocab_size = args["vocab_size"]
        self.embed = nn.Embedding(args['vocab_size'], args['embedding_dim'])
        self.lstm = nn.LSTM(args['embedding_dim'], args['rnn_units'], batch_first=True)
        self.linear = nn.Linear(args['rnn_units'], args['vocab_size'])

    def forward(self, *input, **kwargs):
        embed = self.embed(input)
        feat = self.lstm(embed)[:, 0]
        out = self.linear(feat)
        return out

    def predict(self, *args, seq_len=1, **kwargs):
        """Evalutaion mode prediction"""
        self.eval()
        with torch.no_grad():
            temperature = self.configuration["evaluation"]["temperature"]
            seed = self.configuration["evaluation"]["seed"]
            char2idx = self.configuration["evaluation"]["output_path"] + "{}_char2idx.json".format(
                self.configuration["evaluation"]["antigen"])
            idx2char = {v: k for k, v in char2idx.items()}
            start_seq = np.array([self.char2idx[aa] for aa in seed])
            gen_seq = ''
            for c in range(seq_len):
                output = self.forward(start_seq)
                output = output / temperature
                char_index = Categorical(output).sample()[0, -1]
                gen_seq += idx2char[char_index.cpu().item()]
                start_seq = char_index.unsqueeze(0).unsqueeze(1)

            output_filename = self.configuration["evaluation"]["result_path"] + "{}_generated_sequence.txt"
            with open(output_filename, 'w') as f:
                f.write(gen_seq)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        loaded_state_dict = torch.load(path)
        self.load_state_dict(loaded_state_dict)
        self.eval()

    def fit(self):
        config = self.config

        # Reproducibility of Experiments
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        device = torch.device(f"cuda:{config['gpu_id']}" if config['cuda'] else 'cpu')

        train_loader, eval_loader = MultiModalDataLoader(config)
        trainloss, trainloss_epoch, validloss, testloss_epoch = [], [], [], []

        optim = torch.optim.Adam(self.parameters(),
                                 lr=config['optim']['lr'],
                                 betas=(config['optim']['beta1'],
                                        config['optim']['beta2']))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100, eta_min=0, last_epoch=-1)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(config['epochs']):
            trainloss_epoch, nm_samples = 0, 0
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optim.zero_grad()
                energy = self.forward(batch_x)
                y_score = energy.flatten()
                loss = loss_fn(y_score, batch_y)
                loss.backward()
                trainloss.append(loss.item())
                optim.step()
                trainloss_epoch += trainloss[-1]
                nm_samples += len(batch_x)

            trainloss_epoch.append(trainloss_epoch / nm_samples)
            scheduler.step()

            print(f"Epoch {epoch + 1} Training Loss {trainloss_epoch[epoch]:.16f}")

            if (epoch + 1) % config['save_every'] == 0:
                self.save_model(epoch)

        self.plot_loss()

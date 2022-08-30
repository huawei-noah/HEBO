import torch
import torch.nn as nn
from torch.nn import Module
import math
from model.base import BaseModel
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions.categorical import Categorical
import numpy as np

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(BaseModel, Module):
    def __init__(self, configuration, device):
        BaseModel.__init__(self)
        Module.__init__(self)
        self.configuration = configuration
        self.device = device
        self.create_network()
        args = self.configuration["model"]
        self.n_token = args["n_token"]
        self.d_model = args["d_model"]
        self.n_head = args["n_head"]
        self.d_hid = args["d_hid"]
        self.n_layers = args["n_layers"]
        self.dropout = args["dropout"]


    def create_network(self):
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout).to(self.device)
        encoder_layers = TransformerEncoderLayer(self.d_model, self.n_head, self.d_hid, self.dropout).to(self.device)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.n_layers).to(self.device)
        self.encoder = nn.Embedding(self.n_token, self.d_model)
        self.decoder = nn.Linear(self.d_model, self.n_token)
        self.softmax = nn.Softmax()

    def forward(self, *input, **kwargs):
        src_mask = generate_square_subsequent_mask(input.shape[1])

        src = self.encoder(input) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        output = self.softmax(output)
        output = torch.swapaxes(output, 0, 1)
        return output

    def predict(self, *args, seq_len=1, **kwargs):
        """Evalutaion mode prediction"""
        self.eval()
        with torch.no_grad():
            temperature = self.configuration["evaluation"]["temperature"]
            seed = self.configura/home/derrick/antigenbindingtion["evaluation"]["seed"]
            char2idx = self.configuration["evaluation"]["output_path"] + "{}_char2idx.json".format(self.configuration["evaluation"]["antigen"])
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

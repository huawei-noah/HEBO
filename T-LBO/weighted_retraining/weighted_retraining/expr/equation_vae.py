import collections
import re
from typing import List, Union

import nltk
import numpy as np
import torch
from torch import Tensor

from weighted_retraining.weighted_retraining.expr import eq_grammar, expr_model_pt
from weighted_retraining.weighted_retraining.expr.expr_model_pt import EquationVaeTorch


def tokenize(s):
    funcs = ['sin', 'exp']
    for fn in funcs:
        s = s.replace(fn + '(', fn + ' ')
    s = re.sub(r'([^a-z ])', r' \1 ', s)
    for fn in funcs:
        s = s.replace(fn, fn + '(')
    return s.split()


def pop_or_nothing(S):
    try:
        return S.pop()
    except:
        return 'Nothing'


def prods_to_eq(prods):
    seq = [prods[0].lhs()]
    for prod in prods:
        if str(prod.lhs()) == 'Nothing':
            break
        for ix, s in enumerate(seq):
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix + 1:]
                break
    try:
        return ''.join(seq)
    except:
        return ''


def string_to_one_hot(x_str, data_map, n_entries, max_len):
    """ convert string representation to one-hot representation """

    indices = [np.array([data_map[e] for e in entry], dtype=int)
               for entry in x_str]
    one_hot = np.zeros((len(indices), max_len, n_entries), dtype=np.float32)
    for i in range(len(indices)):
        num_productions = len(indices[i])
        one_hot[i][np.arange(num_productions), indices[i]] = 1.
        one_hot[i][np.arange(num_productions, max_len), -1] = 1.
    return one_hot


class EquationGrammarModelTorch:

    def __init__(self, vae: EquationVaeTorch):
        """ Load the (trained) equation encoder/decoder, grammar model. """
        self._grammar = eq_grammar
        self._model = expr_model_pt
        self.MAX_LEN = 15
        self._productions = self._grammar.GCFG.productions()
        self._prod_map = {}
        for ix, prod in enumerate(self._productions):
            self._prod_map[prod] = ix
        self._parser = nltk.ChartParser(self._grammar.GCFG)
        self._tokenize = tokenize
        self._n_chars = len(self._productions)
        self._lhs_map = {}
        for ix, lhs in enumerate(self._grammar.lhs_list):
            self._lhs_map[lhs] = ix

        self.vae: EquationVaeTorch = vae

    @property
    def tkwargs(self):
        return dict(device=self.vae.device, dtype=self.vae.dtype)

    def encode(self, smiles: List[str]) -> np.ndarray:
        """ Encode a list of smiles strings into the latent space """
        one_hot = self.smiles_to_one_hot(smiles)
        self.one_hot = torch.tensor(one_hot).to(**self.tkwargs)
        return self.vae.encode_to_params(self.one_hot)[0].cpu().detach().numpy()

    def smiles_to_one_hot(self, smiles: List[str]):
        """ convert smiles to one-hot vectors """
        assert type(smiles) == list
        try:
            tokens = list(map(self._tokenize, smiles))
        except AttributeError as e:
            raise AttributeError(f"{smiles}  --- " + ' | '.join(map(str, set(map(type, smiles))))) from e
        parse_trees = [next(self._parser.parse(t)) for t in tokens]
        productions_seq = [tree.productions() for tree in parse_trees]
        return string_to_one_hot(productions_seq, self._prod_map, self._n_chars, self.MAX_LEN)

    def _sample_using_masks(self, unmasked):
        """ Samples a one-hot vector, masking at each timestep.
            This is an implementation of Algorithm ? in the paper. """
        eps = 1e-100
        X_hat = np.zeros_like(unmasked)

        # Create a stack for each input in the batch
        S = np.empty((unmasked.shape[0],), dtype=object)
        for ix in range(S.shape[0]):
            S[ix] = [str(self._grammar.start_index)]

        # Loop over time axis, sampling values and updating masks
        for t in range(unmasked.shape[1]):
            next_nonterminal = [self._lhs_map[pop_or_nothing(a)] for a in S]
            mask = self._grammar.masks[next_nonterminal]
            masked_output = np.exp(unmasked[:, t, :]) * mask + eps
            sampled_output = np.argmax(np.random.gumbel(
                size=masked_output.shape) + np.log(masked_output), axis=-1)
            X_hat[np.arange(unmasked.shape[0]), t, sampled_output] = 1.0

            # Identify non-terminals in RHS of selected production, and
            # push them onto the stack in reverse order
            rhs = [
                [a for a in self._productions[i].rhs() if (type(a) == nltk.grammar.Nonterminal) and (str(a) != 'None')]
                for i in sampled_output]
            for ix in range(S.shape[0]):
                S[ix].extend(list(map(str, rhs[ix]))[::-1])
        return X_hat  # , ln_p

    def decode(self, z: Union[np.ndarray, Tensor]):
        """ Sample from the grammar decoder """
        assert z.ndim == 2, z.shape
        if not isinstance(z, Tensor):
            z = torch.tensor(z)
        z_torch = z.to(**self.tkwargs)
        unmasked = self.vae.decode_deterministic(z_torch).cpu().detach().numpy()
        X_hat = self._sample_using_masks(unmasked)
        # Convert from one-hot to sequence of production rules
        prod_seq = [[self._productions[X_hat[index, t].argmax()]
                     for t in range(X_hat.shape[1])]
                    for index in range(X_hat.shape[0])]
        return [prods_to_eq(prods) for prods in prod_seq]

    def decode_from_latent_space(self, zs: Union[Tensor, np.ndarray], n_decode_attempts=1) -> np.ndarray:
        """ decode from latents to inputs and set all invalid inputs to None """

        # decode equations and replace all empty ones (i.e. '') by None
        if not isinstance(zs, Tensor):
            zs_torch = torch.tensor(zs).to(**self.tkwargs)
        else:
            zs_torch = zs
        decoded_equations = [self.decode(zs_torch) for _ in range(n_decode_attempts)]
        valid_equations = []
        for i in range(n_decode_attempts):
            valid_equations.append([])
            for j in range(zs.shape[0]):
                eq = np.array([decoded_equations[i][j]]).astype('str')[0]
                valid_equations[i].append(None if eq == '' else eq)

        # if the different decoding attempts yielded different equations, pick the majority
        valid_equations = np.array(valid_equations)
        final_equations = []
        for i in range(zs.shape[0]):
            aux = collections.Counter(valid_equations[~np.equal(valid_equations[:, i], None), i])
            eq = list(aux.items())[np.argmax(list(aux.values()))][0] if len(aux) > 0 else None
            final_equations.append(eq)

        return np.array(final_equations)

    def eval(self):
        self.vae.eval()

    def train(self):
        self.vae.train()

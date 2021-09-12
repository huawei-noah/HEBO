import torch
import torch.nn as nn
import torch.nn.functional as F
from .mol_tree import Vocab, MolTree, MolTreeNode
from .nnutils import GRU
from .chemutils import enum_assemble, set_atommap
import copy

MAX_NB = 15
MAX_DECODE_LEN = 100


class JTNNDecoder(nn.Module):
    def __init__(self, vocab, hidden_size, latent_tree_size, embedding):
        super(JTNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab
        self.embedding = embedding

        # GRU Weights
        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)

        # Word Prediction Weights
        self.W = nn.Linear(hidden_size + latent_tree_size, hidden_size)

        # Stop Prediction Weights
        self.U = nn.Linear(hidden_size + latent_tree_size, hidden_size)
        self.U_i = nn.Linear(2 * hidden_size, hidden_size)
        self.U_i_relu = True  # flag for compatibility with prev models

        # Output Weights
        self.W_o = nn.Linear(hidden_size, self.vocab_size)
        self.U_o = nn.Linear(hidden_size, 1)

        # Loss Functions
        self.pred_loss = nn.CrossEntropyLoss(reduction="sum")
        self.stop_loss = nn.BCEWithLogitsLoss(reduction="sum")
        self.batch_pred_loss = nn.CrossEntropyLoss(reduction="none")
        self.batch_stop_loss = nn.BCEWithLogitsLoss(reduction="none")


    def aggregate(self, hiddens, contexts, x_tree_vecs, mode):
        if mode == "word":
            V, V_o = self.W, self.W_o
        elif mode == "stop":
            V, V_o = self.U, self.U_o
        else:
            raise ValueError("aggregate mode is wrong")

        tree_contexts = x_tree_vecs.index_select(0, contexts)
        input_vec = torch.cat([hiddens, tree_contexts], dim=-1)
        output_vec = F.relu(V(input_vec))
        return V_o(output_vec)

    def forward(self, mol_batch, x_tree_vecs, return_batch: bool = False):
        pred_hiddens, pred_contexts, pred_targets = [], [], []
        stop_hiddens, stop_contexts, stop_targets = [], [], []
        traces = []
        for mol_tree in mol_batch:
            s = []
            dfs(s, mol_tree.nodes[0], -1)
            traces.append(s)
            for node in mol_tree.nodes:
                node.neighbors = []

        # Predict Root
        batch_size = len(mol_batch)
        pred_hiddens.append(
            torch.zeros(
                len(mol_batch), self.hidden_size, device=self.embedding.weight.device
            )
        )
        pred_targets.extend([mol_tree.nodes[0].wid for mol_tree in mol_batch])
        pred_contexts.append(
            torch.as_tensor(
                list(range(batch_size)),
                dtype=torch.long,
                device=self.embedding.weight.device,
            )
        )
        max_iter = max([len(tr) for tr in traces])
        padding = torch.zeros(self.hidden_size, device=self.embedding.weight.device)
        h = {}

        for t in range(max_iter):
            prop_list = []
            batch_list = []
            for i, plist in enumerate(traces):
                if t < len(plist):
                    prop_list.append(plist[t])
                    batch_list.append(i)

            cur_x = []
            cur_h_nei, cur_o_nei = [], []

            for node_x, real_y, _ in prop_list:
                # Neighbors for message passing (target not included)
                cur_nei = [
                    h[(node_y.idx, node_x.idx)]
                    for node_y in node_x.neighbors
                    if node_y.idx != real_y.idx
                ]
                pad_len = MAX_NB - len(cur_nei)
                cur_h_nei.extend(cur_nei)
                cur_h_nei.extend([padding] * pad_len)

                # Neighbors for stop prediction (all neighbors)
                cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
                pad_len = MAX_NB - len(cur_nei)
                cur_o_nei.extend(cur_nei)
                cur_o_nei.extend([padding] * pad_len)

                # Current clique embedding
                cur_x.append(node_x.wid)

            # Clique embedding
            cur_x = torch.as_tensor(
                cur_x, dtype=torch.long, device=self.embedding.weight.device
            )
            cur_x = self.embedding(cur_x)

            # Message passing
            cur_h_nei = torch.stack(cur_h_nei, dim=0).view(-1, MAX_NB, self.hidden_size)
            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)

            # Node Aggregate
            cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1, MAX_NB, self.hidden_size)
            cur_o = cur_o_nei.sum(dim=1)

            # Gather targets
            pred_target, pred_list = [], []
            stop_target = []
            for i, m in enumerate(prop_list):
                node_x, node_y, direction = m
                x, y = node_x.idx, node_y.idx
                h[(x, y)] = new_h[i]
                node_y.neighbors.append(node_x)
                if direction == 1:
                    pred_target.append(node_y.wid)
                    pred_list.append(i)
                stop_target.append(direction)

            # Hidden states for stop prediction
            cur_batch = torch.as_tensor(
                batch_list, dtype=torch.long, device=self.embedding.weight.device
            )
            stop_hidden = torch.cat([cur_x, cur_o], dim=1)
            stop_hiddens.append(stop_hidden)
            stop_contexts.append(cur_batch)
            stop_targets.extend(stop_target)

            # Hidden states for clique prediction
            if len(pred_list) > 0:
                batch_list = [batch_list[i] for i in pred_list]
                cur_batch = torch.as_tensor(
                    batch_list, dtype=torch.long, device=self.embedding.weight.device
                )
                pred_contexts.append(cur_batch)

                cur_pred = torch.as_tensor(
                    pred_list, dtype=torch.long, device=self.embedding.weight.device
                )
                pred_hiddens.append(new_h.index_select(0, cur_pred))
                pred_targets.extend(pred_target)

        # Last stop at root
        cur_x, cur_o_nei = [], []
        for mol_tree in mol_batch:
            node_x = mol_tree.nodes[0]
            cur_x.append(node_x.wid)
            cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
            pad_len = MAX_NB - len(cur_nei)
            cur_o_nei.extend(cur_nei)
            cur_o_nei.extend([padding] * pad_len)

        cur_x = torch.as_tensor(
            cur_x, dtype=torch.long, device=self.embedding.weight.device
        )
        cur_x = self.embedding(cur_x)
        cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1, MAX_NB, self.hidden_size)
        cur_o = cur_o_nei.sum(dim=1)

        stop_hidden = torch.cat([cur_x, cur_o], dim=1)
        stop_hiddens.append(stop_hidden)
        stop_contexts.append(
            torch.as_tensor(
                list(range(batch_size)),
                dtype=torch.long,
                device=self.embedding.weight.device,
            )
        )
        stop_targets.extend([0] * len(mol_batch))

        # Predict next clique
        pred_contexts = torch.cat(pred_contexts, dim=0)
        pred_hiddens = torch.cat(pred_hiddens, dim=0)
        pred_scores = self.aggregate(pred_hiddens, pred_contexts, x_tree_vecs, "word")
        pred_targets = torch.as_tensor(
            pred_targets, device=self.embedding.weight.device, dtype=torch.long
        )

        if return_batch:
            pred_loss = self.batch_pred_loss(pred_scores, pred_targets)
        else:
            pred_loss = self.pred_loss(pred_scores, pred_targets) / len(mol_batch)

        _, preds = torch.max(pred_scores, dim=1)
        pred_acc = torch.eq(preds, pred_targets).float()
        pred_acc = torch.sum(pred_acc) / pred_targets.nelement()

        # Predict stop
        stop_contexts = torch.cat(stop_contexts, dim=0)
        stop_hiddens = torch.cat(stop_hiddens, dim=0)
        if self.U_i_relu:
            stop_hiddens = F.relu(self.U_i(stop_hiddens))
        else:
            stop_hiddens = self.U_i(stop_hiddens)
        stop_scores = self.aggregate(stop_hiddens, stop_contexts, x_tree_vecs, "stop")
        stop_scores = stop_scores.squeeze(-1)
        stop_targets = torch.as_tensor(
            stop_targets, device=self.embedding.weight.device, dtype=torch.float
        )

        if return_batch:
            stop_loss = self.batch_stop_loss(stop_scores, stop_targets)
        else:
            stop_loss = self.stop_loss(stop_scores, stop_targets) / len(mol_batch)

        stops = torch.ge(stop_scores, 0).float()
        stop_acc = torch.eq(stops, stop_targets).float()
        stop_acc = torch.sum(stop_acc) / stop_targets.nelement()

        return pred_loss, stop_loss, pred_acc.item(), stop_acc.item()

    def decode(self, x_tree_vecs, prob_decode):
        assert x_tree_vecs.size(0) == 1

        stack = []
        init_hiddens = torch.zeros(
            1, self.hidden_size, device=self.embedding.weight.device
        )
        zero_pad = torch.zeros(
            1, 1, self.hidden_size, device=self.embedding.weight.device
        )
        contexts = torch.as_tensor(
            1, dtype=torch.long, device=self.embedding.weight.device
        ).zero_()

        # Root Prediction
        root_score = self.aggregate(init_hiddens, contexts, x_tree_vecs, "word")
        _, root_wid = torch.max(root_score, dim=1)
        root_wid = root_wid.item()

        root = MolTreeNode(self.vocab.get_smiles(root_wid))
        root.wid = root_wid
        root.idx = 0
        stack.append((root, self.vocab.get_slots(root.wid)))

        all_nodes = [root]
        h = {}
        for step in range(MAX_DECODE_LEN):
            node_x, fa_slot = stack[-1]
            cur_h_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
            if len(cur_h_nei) > 0:
                cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1, -1, self.hidden_size)
            else:
                cur_h_nei = zero_pad

            cur_x = torch.as_tensor(
                [node_x.wid], dtype=torch.long, device=self.embedding.weight.device
            )
            cur_x = self.embedding(cur_x)

            # Predict stop
            cur_h = cur_h_nei.sum(dim=1)
            stop_hiddens = torch.cat([cur_x, cur_h], dim=1)
            if self.U_i_relu:
                stop_hiddens = F.relu(self.U_i(stop_hiddens))
            else:
                stop_hiddens = self.U_i(stop_hiddens)
            stop_score = self.aggregate(stop_hiddens, contexts, x_tree_vecs, "stop")

            if prob_decode:
                backtrack = torch.bernoulli(torch.sigmoid(stop_score)).item() == 0
            else:
                backtrack = stop_score.item() < 0

            if not backtrack:  # Forward: Predict next clique
                new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                pred_score = self.aggregate(new_h, contexts, x_tree_vecs, "word")

                if prob_decode:
                    sort_wid = torch.multinomial(
                        F.softmax(pred_score, dim=1).squeeze(), 5
                    )
                else:
                    _, sort_wid = torch.sort(pred_score, dim=1, descending=True)
                    sort_wid = sort_wid.data.squeeze()

                next_wid = None
                for wid in sort_wid[:5]:
                    slots = self.vocab.get_slots(wid)
                    node_y = MolTreeNode(self.vocab.get_smiles(wid))
                    if have_slots(fa_slot, slots) and can_assemble(node_x, node_y):
                        next_wid = wid
                        next_slots = slots
                        break

                if next_wid is None:
                    backtrack = True  # No more children can be added
                else:
                    node_y = MolTreeNode(self.vocab.get_smiles(next_wid))
                    node_y.wid = next_wid
                    node_y.idx = len(all_nodes)
                    node_y.neighbors.append(node_x)
                    h[(node_x.idx, node_y.idx)] = new_h[0]
                    stack.append((node_y, next_slots))
                    all_nodes.append(node_y)

            if backtrack:  # Backtrack, use if instead of else
                if len(stack) == 1:
                    break  # At root, terminate

                node_fa, _ = stack[-2]
                cur_h_nei = [
                    h[(node_y.idx, node_x.idx)]
                    for node_y in node_x.neighbors
                    if node_y.idx != node_fa.idx
                ]
                if len(cur_h_nei) > 0:
                    cur_h_nei = torch.stack(cur_h_nei, dim=0).view(
                        1, -1, self.hidden_size
                    )
                else:
                    cur_h_nei = zero_pad

                new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                h[(node_x.idx, node_fa.idx)] = new_h[0]
                node_fa.neighbors.append(node_x)
                stack.pop()

        return root, all_nodes


"""
Helper Functions:
"""


def dfs(stack, x, fa_idx):
    for y in x.neighbors:
        if y.idx == fa_idx:
            continue
        stack.append((x, y, 1))
        dfs(stack, y, x.idx)
        stack.append((y, x, 0))


def have_slots(fa_slots, ch_slots):
    if len(fa_slots) > 2 and len(ch_slots) > 2:
        return True
    matches = []
    for i, s1 in enumerate(fa_slots):
        a1, c1, h1 = s1
        for j, s2 in enumerate(ch_slots):
            a2, c2, h2 = s2
            if a1 == a2 and c1 == c2 and (a1 != "C" or h1 + h2 >= 4):
                matches.append((i, j))

    if len(matches) == 0:
        return False

    fa_match, ch_match = list(zip(*matches))
    if (
        len(set(fa_match)) == 1 and 1 < len(fa_slots) <= 2
    ):  # never remove atom from ring
        fa_slots.pop(fa_match[0])
    if (
        len(set(ch_match)) == 1 and 1 < len(ch_slots) <= 2
    ):  # never remove atom from ring
        ch_slots.pop(ch_match[0])

    return True


def can_assemble(node_x, node_y):
    node_x.nid = 1
    node_x.is_leaf = False
    set_atommap(node_x.mol, node_x.nid)

    neis = node_x.neighbors + [node_y]
    for i, nei in enumerate(neis):
        nei.nid = i + 2
        nei.is_leaf = len(nei.neighbors) <= 1
        if nei.is_leaf:
            set_atommap(nei.mol, 0)
        else:
            set_atommap(nei.mol, nei.nid)

    neighbors = [nei for nei in neis if nei.mol.GetNumAtoms() > 1]
    neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
    singletons = [nei for nei in neis if nei.mol.GetNumAtoms() == 1]
    neighbors = singletons + neighbors
    cands, aroma_scores = enum_assemble(node_x, neighbors)
    return len(cands) > 0  # and sum(aroma_scores) >= 0


if __name__ == "__main__":
    smiles = [
        "O=C1[C@@H]2C=C[C@@H](C=CC2)C1(c1ccccc1)c1ccccc1",
        "O=C([O-])CC[C@@]12CCCC[C@]1(O)OC(=O)CC2",
        "ON=C1C[C@H]2CC3(C[C@@H](C1)c1ccccc12)OCCO3",
        "C[C@H]1CC(=O)[C@H]2[C@@]3(O)C(=O)c4cccc(O)c4[C@@H]4O[C@@]43[C@@H](O)C[C@]2(O)C1",
        "Cc1cc(NC(=O)CSc2nnc3c4ccccc4n(C)c3n2)ccc1Br",
        "CC(C)(C)c1ccc(C(=O)N[C@H]2CCN3CCCc4cccc2c43)cc1",
        "O=c1c2ccc3c(=O)n(-c4nccs4)c(=O)c4ccc(c(=O)n1-c1nccs1)c2c34",
        "O=C(N1CCc2c(F)ccc(F)c2C1)C1(O)Cc2ccccc2C1",
    ]
    for s in smiles:
        print(s)
        tree = MolTree(s)
        for i, node in enumerate(tree.nodes):
            node.idx = i

        stack = []
        dfs(stack, tree.nodes[0], -1)
        for x, y, d in stack:
            print(x.smiles, y.smiles, d)
        print("------------------------------")


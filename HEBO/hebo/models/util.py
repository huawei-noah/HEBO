# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import random

import networkx as nx
import torch
from disjoint_set import DisjointSet
from torch import FloatTensor, LongTensor, nn


def filter_nan(x : FloatTensor, xe : LongTensor, y : FloatTensor, keep_rule = 'any') -> (FloatTensor, LongTensor, FloatTensor):
    assert x  is None or torch.isfinite(x).all()
    assert xe is None or torch.isfinite(xe).all()
    assert torch.isfinite(y).any(), "No valid data in the dataset"

    if keep_rule == 'any':
        valid_id = torch.isfinite(y).any(dim = 1)
    else:
        valid_id = torch.isfinite(y).all(dim = 1)
    x_filtered  = x[valid_id]  if x  is not None else None
    xe_filtered = xe[valid_id] if xe is not None else None
    y_filtered  = y[valid_id]
    return x_filtered, xe_filtered, y_filtered

def construct_hidden(dim, num_layers, num_hiddens, act = nn.ReLU()) -> nn.Module:
    layers = [nn.Linear(dim, num_hiddens), act]
    for i in range(num_layers - 1):
        layers.append(nn.Linear(num_hiddens, num_hiddens))
        layers.append(act)
    return nn.Sequential(*layers)

def get_random_graph(size, E):
    graph = nx.empty_graph(size)
    disjoint_set = DisjointSet()
    connections_made = 0
    while connections_made < min(size - 1, max(int(E * size), 1)):
        edge_in = random.randint(0, size - 1)
        edge_out = random.randint(0, size - 1)

        if edge_in == edge_out or disjoint_set.connected(edge_out, edge_in):
            continue
        else:
            connections_made += 1
            graph.add_edge(edge_in, edge_out)
            disjoint_set.union(edge_in, edge_out)

        return list(nx.find_cliques(graph))
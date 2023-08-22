# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from typing import List

import numpy as np
import torch

from mcbo.search_space import SearchSpace


def laplacian_eigen_decomposition(search_space: SearchSpace, device: torch.device) -> (
        np.ndarray, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]):
    """
    Function used to compute the eigen decomposition of the Laplacian of a combinatorial graph.

    Args:
        search_space: Search space used to build the combinatorial graph.

    Returns:
        n_vertices:
        adjacency_matrix_list:
        fourier_frequency_list:
        fourier_basis_list:
    """

    assert search_space.num_nominal + search_space.num_ordinal == search_space.num_params, \
        'We can only build a combinatorial graph for nominal and ordinal variables'

    n_vertices = []
    adjacency_matrix_list = []
    fourier_frequency_list = []
    fourier_basis_list = []
    dtype = search_space.dtype

    for i, param in enumerate(search_space.params):
        n_v = search_space.params[param].ub - search_space.params[param].lb + 1

        if search_space.params[param].is_ordinal:
            adj_mat = torch.diag(torch.ones(n_v - 1, dtype=dtype, device=device), -1) + torch.diag(
                torch.ones(n_v - 1, dtype=dtype, device=device), 1)
            laplacian = (torch.diag(torch.sum(adj_mat, dim=0)) - adj_mat)

        elif search_space.params[param].is_nominal:
            adj_mat = torch.ones((n_v, n_v), dtype=dtype, device=device).fill_diagonal_(0)
            laplacian = (torch.diag(torch.sum(adj_mat, dim=0)) - adj_mat) / n_v

        eigval, eigvec = torch.linalg.eigh(laplacian, UPLO='U')

        n_vertices.append(n_v)
        adjacency_matrix_list.append(adj_mat)
        fourier_frequency_list.append(eigval)
        fourier_basis_list.append(eigvec)

    return np.array(n_vertices), adjacency_matrix_list, fourier_frequency_list, fourier_basis_list


def cartesian_neighbors(x: torch.Tensor, edge_mat_list: List[torch.Tensor]) -> torch.Tensor:
    """
    For given vertices, it returns all neighboring vertices on cartesian product of the graphs given by edge_mat_list

    Args:
        x: 1D Tensor
        edge_mat_list: list of adjacency

    Returns:
         2d tensor in which each row is 1-hamming distance far from x
    """
    neighbor_list = []
    for i in range(len(edge_mat_list)):
        nbd_i_elm = edge_mat_list[i][x[i]].nonzero(as_tuple=False).squeeze(1)
        nbd_i = x.repeat((nbd_i_elm.numel(), 1))
        nbd_i[:, i] = nbd_i_elm
        neighbor_list.append(nbd_i)

    return torch.cat(neighbor_list, dim=0)


def cartesian_neighbors_center_attracted(x: torch.Tensor, edge_mat_list: List[torch.Tensor],
                                         x_center: torch.Tensor) -> torch.Tensor:
    """
    For given vertices, it returns all neighboring vertices on cartesian product of the graphs given by edge_mat_list

    Args:
        x: 1D Tensor
        edge_mat_list: list of adjacency
        x_center: to be selected, the neighbor must have hamming(x_neigh, x_center) <= hamming(x, x_center)

    Returns:
         2d tensor in which each row is 1-hamming distance far from x
    """
    neighbor_list = []
    for i in range(len(edge_mat_list)):
        if x_center[i] == x[i]:
            # cannot change this dim
            continue
        nbd_i_elm = edge_mat_list[i][x[i]].nonzero(as_tuple=False).squeeze(1)  # get indices of the connected cats
        nbd_i = x.repeat((nbd_i_elm.numel(), 1))
        nbd_i[:, i] = nbd_i_elm
        neighbor_list.append(nbd_i)

    return torch.cat(neighbor_list, dim=0)

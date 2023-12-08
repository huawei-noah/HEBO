# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided
# that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
# following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from typing import List, Tuple

import numpy as np
from scipy.ndimage import rotate

from mcbo.search_space import SearchSpace


def t_square_fractal(side: int, coefs: List[Tuple[float]], offset_ratio: float,
                     fill_center: bool, negative: bool,
                     rot: float, quarter_rot: int):
    """ Generate a t-square fractal of given side

    :param side: side of the grid on which to generate the fractal
    :param coefs: list of tuples (a_i, b_i, c_i) characterising the properties of the fractal at each scale `i`.
    :param offset_ratio: the fractal is at the center of the grid, and a margin is left on the border of the grid. The width
    of this border is `offset_ratio * side / 2`
    :param fill_center: one of the fractal characteristics (can be active or not)
    :param negative: whether to return `grid` or `1 - grid`
    :param rot: rotation parameter -> applied rotation of `rot x 90` degrees
    :param quarter_rot: rotation parameter -> applied rotation of each quadrant of `quarter_rot x 90` degrees
    """
    true_side = side
    assert 0 <= offset_ratio <= 1
    edge = round(side / 2 * offset_ratio)
    side = true_side - 2 * edge
    if side == 0:
        full_grid = np.zeros((true_side, true_side))
    else:
        grid = np.ones((side, side))
        if len(coefs) > 0 and side > 4:

            a, b, c = coefs.pop(0)

            assert 0 <= a <= 1
            assert 0 <= b <= 1
            assert 0 <= c <= 1
            assert 0 <= quarter_rot <= 3

            to_remove_w1 = (side - 4) * a
            if side % 2 == 0:
                to_remove_w1 = round(to_remove_w1 / 2.) * 2
            else:
                if to_remove_w1 <= .5:
                    to_remove_w1 = 0
                else:
                    to_remove_w1 = round((to_remove_w1 - 1) / 2.) * 2 + 1

            subside = (side - to_remove_w1) // 2
            if to_remove_w1 == 0:
                to_remove_w2 = 0
            else:
                to_remove_w2 = round(b * (subside - 1))

            if to_remove_w1 * to_remove_w2 > 0:
                grid[subside:subside + to_remove_w1, :to_remove_w2] = 0
                grid[subside:subside + to_remove_w1, -to_remove_w2:] = 0
                grid.T[subside:subside + to_remove_w1, :to_remove_w2] = 0
                grid.T[subside:subside + to_remove_w1, -to_remove_w2:] = 0

            subpattern = t_square_fractal(side=subside, coefs=coefs, offset_ratio=0, fill_center=fill_center,
                                          negative=False, rot=0, quarter_rot=0)
            grid[:subside, :subside] = subpattern
            grid[:subside, -subside:] = subpattern
            grid[-subside:, -subside:] = subpattern
            grid[-subside:, :subside] = subpattern

            if fill_center:
                grid[to_remove_w2:side - to_remove_w2, to_remove_w2:side - to_remove_w2] = 1
            if c > 0:
                central_square_side = side - to_remove_w2 * 2
                to_remove_square = (central_square_side) * c
                if central_square_side % 2 == 0:
                    to_remove_square = round(to_remove_square / 2.) * 2
                else:
                    if to_remove_square <= .5:
                        to_remove_square = 0
                    else:
                        to_remove_square = round((to_remove_square - 1) / 2.) * 2 + 1
                if to_remove_square > 0:
                    square_edge = (central_square_side - to_remove_square) // 2
                    grid[to_remove_w2 + square_edge:-to_remove_w2 - square_edge,
                    to_remove_w2 + square_edge:-to_remove_w2 - square_edge] = 0

        if edge == 0:
            full_grid = grid
        else:
            full_grid = np.zeros((true_side, true_side))
            full_grid[edge:-edge, edge:-edge] = grid
    if negative:
        full_grid = 1 - full_grid
    full_grid = np.round(np.clip(rotate(full_grid, angle=rot * 90, reshape=False), 0, 1))
    if quarter_rot > 0:
        assert true_side % 2 == 0
        quarter_side = true_side // 2
        pattern = full_grid[-quarter_side:, -quarter_side:]
        pattern = np.rot90(pattern, k=quarter_rot)

        (height, width) = pattern.shape
        full_grid = np.zeros((pattern.shape[0] * 2, pattern.shape[1] * 2))
        full_grid[0:height, 0:width] = np.rot90(np.rot90(pattern))
        full_grid[0:height, width:] = np.rot90(pattern)
        full_grid[height:, 0:width] = np.rot90(np.rot90(np.rot90(pattern)))
        full_grid[height:, width:] = pattern

    return full_grid


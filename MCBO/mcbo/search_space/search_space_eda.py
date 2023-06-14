from typing import List, Dict

import numpy as np
import torch
import pandas as pd

from mcbo.search_space import SearchSpace
from mcbo.tasks.eda_seq_opt.utils.utils_operators import get_seq_operators_pattern


class SearchSpaceEDA(SearchSpace):
    def __init__(self, params: List[dict], dtype: torch.dtype, seq_operators_pattern_id: str,
                 op_ind_per_type_dic: Dict[int, List[int]]):
        super(SearchSpaceEDA, self).__init__(params=params, dtype=dtype)
        self.seq_operators_pattern_id = seq_operators_pattern_id
        self.op_ind_per_type_dic = op_ind_per_type_dic

        self.op_to_type_dic: Dict[int, int] = {}  # indicates for each op what is its op_type
        for op_type, op_list in self.op_ind_per_type_dic.items():
            for op in op_list:
                assert op not in self.op_to_type_dic, (op, op_type, self.op_to_type_dic[op])
                self.op_to_type_dic[op] = op_type

        self.seq_operators_pattern = get_seq_operators_pattern(
            seq_operators_pattern=self.seq_operators_pattern_id)
        self.seq_len = len(self.seq_operators_pattern)

    def get_transformed_mutation_cand_values(self, transformed_x: torch.Tensor, dim: int) -> np.ndarray:
        """ Get the values that could replace transformed_x[dim] """
        assert transformed_x.shape == (self.num_dims,), (transformed_x.shape, self.num_dims)
        current_seq_inverse = self.inverse_transform(transformed_x.unsqueeze(0).detach()).values[0, :self.seq_len]
        valid_original_cats: List[int] = self.seq_operators_pattern.get_valid_ops_at_dim(
            current_seq=current_seq_inverse, dim=dim, op_ind_per_type_dic=self.op_ind_per_type_dic,
            op_to_type_dic=self.op_to_type_dic
        )
        assert len(valid_original_cats) >= 0
        return self.params[self.param_names[dim]].transform(np.array(valid_original_cats)).cpu().numpy()

    def sample(self, num_samples=1) -> pd.DataFrame:
        """
        num_samples: number of desired samples
        """
        df_suggest = super(SearchSpaceEDA, self).sample(num_samples=num_samples)
        if not self.seq_operators_pattern.contains_free:
            return df_suggest
        for i in range(num_samples):
            # correct the samples
            df_suggest.iloc[i, :self.seq_len] = self.seq_operators_pattern.sample(
                op_ind_per_type_dic=self.op_ind_per_type_dic)
        return df_suggest

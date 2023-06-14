from typing import List, Dict, Any, Optional

import numpy as np

from mcbo.search_space import SearchSpace
from mcbo.tasks.eda_seq_opt.utils.utils_operators import OperatorSpace, SeqOperatorsPattern, get_operator_space
from mcbo.tasks.eda_seq_opt.utils.utils_operators_hyp import OperatorHypSpace, get_operator_hyperparms_space


class OptimHyperSpace:

    def __init__(self, operator_space: OperatorSpace, seq_operators_pattern: SeqOperatorsPattern,
                 operator_hyperparams_space: Optional[OperatorHypSpace]):
        self.operator_space = operator_space
        self.seq_operators_pattern = seq_operators_pattern
        self.operator_hyperparams_space = operator_hyperparams_space
        self.op_ind_per_type_dic = {}
        self.search_space_params: List[Dict[str, Any]] = self._get_search_space_params()

    def _get_search_space_params(self) -> List[Dict[str, Any]]:
        param_space: List[Dict[str, Any]] = []
        ind_values: List[List[int]] = self.get_ind_values()
        for i in range(len(self.seq_operators_pattern)):  # sequence of operators
            param_space.append({'name': f'algo_{i + 1}', 'type': 'nominal',
                                'categories': ind_values[i]})

        # get search space for the hyperparameters
        if self.operator_hyperparams_space:
            for i_algo, algo in enumerate(self.operator_space.all_operators):
                param_space.extend(self.operator_hyperparams_space.all_hyps[algo.op_id])

        return param_space

    @property
    def seq_len(self) -> int:
        return len(self.seq_operators_pattern)

    @property
    def search_space_len(self) -> int:
        return len(self.search_space_params)

    def get_ind_values(self) -> List[List[int]]:
        """
        Return for each dim the list of the valid operator indices in the flattened operator space
        """
        ind_values = []
        for i, op_type in enumerate(self.seq_operators_pattern.pattern):
            ind_values.append([])
            valid_types = self.seq_operators_pattern.get_valid_types(op_type=op_type)
            aux_ind = 0
            if 0 in valid_types:
                if 0 not in self.op_ind_per_type_dic:
                    self.op_ind_per_type_dic[0] = list(
                        range(aux_ind, len(self.operator_space.pre_mapping_operators)))
                ind_values[-1].extend(self.op_ind_per_type_dic[0])
            aux_ind += len(self.operator_space.pre_mapping_operators)
            if 1 in valid_types:
                if 1 not in self.op_ind_per_type_dic:
                    self.op_ind_per_type_dic[1] = list(
                        range(aux_ind, aux_ind + len(self.operator_space.mapping_operators)))
                ind_values[-1].extend(self.op_ind_per_type_dic[1])
            aux_ind += len(self.operator_space.mapping_operators)
            if 2 in valid_types:
                assert len(
                    self.operator_space.post_mapping_operators) > 0, "Pattern contains post-mapping optim step, " \
                                                                     "but no operator in operator space"
                if 2 not in self.op_ind_per_type_dic:
                    self.op_ind_per_type_dic[2] = list(
                        range(aux_ind, aux_ind + len(self.operator_space.post_mapping_operators)))
                ind_values[-1].extend(self.op_ind_per_type_dic[2])
            assert len(ind_values[-1]) > 1, f"Should have more than one category, {op_type} {valid_types}"
        return ind_values

    def get_num_values_per_dim(self) -> np.ndarray:
        """
        Return for each dim of the flattened operator space the number of valid operators
        """
        num_operators = np.zeros(len(self.seq_operators_pattern), ).astype(int)
        for i, v in enumerate(self.seq_operators_pattern.pattern):
            if v == 0:
                num_operators[i] = len(self.operator_space.pre_mapping_operators)
                continue
            if v == 1:
                num_operators[i] = len(self.operator_space.mapping_operators)
                continue
            if v == 2:
                assert len(
                    self.operator_space.post_mapping_operators) > 0, "Pattern contains post-mapping optim step, " \
                                                                     "but no operator in operator space"
                num_operators[i] = len(self.operator_space.post_mapping_operators)
                continue
            raise ValueError(v)
        return num_operators

    def get_offset_operator_ind_per_dim(self) -> np.ndarray:
        """
        Return for each dim of the flattened operator space the offset to add to [0, num_categories - 1] to get
        the operator in all_operators
        """
        offset = np.zeros(len(self.seq_operators_pattern)).astype(int)
        for i, v in enumerate(self.seq_operators_pattern.pattern):
            aux_ind = 0
            if v == 0:
                offset[i] = 0
                continue
            aux_ind += len(self.operator_space.pre_mapping_operators)
            if v == 1:
                offset[i] = aux_ind
                continue
            aux_ind += len(self.operator_space.mapping_operators)
            if v == 2:
                assert len(
                    self.operator_space.post_mapping_operators) > 0, "Pattern contains post-mapping optim step, " \
                                                                     "but no operator in operator space"
                offset[i] = aux_ind
                continue
            raise ValueError(v)
        return offset


class EDASearchSpaceIndexManager:

    def __init__(self, seq_len: int, search_space: SearchSpace, operator_space_id: str,
                 operator_hyperparams_space_id: str):
        self.seq_len = seq_len

        # dimensions corresponding to sequence of operators
        self.seq_dims = np.arange(self.seq_len)

        # other nominal dims
        self.nominal_non_seq_dims: np.ndarray = np.array(
            [nominal_dim for nominal_dim in search_space.nominal_dims if nominal_dim not in self.seq_dims])
        self.params_cont_dims: List[np.ndarray] = []
        self.params_disc_dims: List[np.ndarray] = []
        self.params_numeric_dims: List[np.ndarray] = []
        self.params_nominal_dims: List[np.ndarray] = []

        self.nominal_dim_to_choices: Dict[int, np.ndarray] = {
            search_space.nominal_dims[i]: np.arange(search_space.nominal_lb[i], search_space.nominal_ub[i] + 1) for i in
            range(len(search_space.nominal_dims))
        }

        hyperparams_space = get_operator_hyperparms_space(operator_hyperparams_space_id=operator_hyperparams_space_id)
        actual_ind = seq_len
        for algo in get_operator_space(
                operator_space_id=operator_space_id).all_operators:  # the first dims are for the sequence of operators
            cont_dims = []
            disc_dims = []
            numeric_dims = []
            nominal_dims = []
            for algo_param in hyperparams_space.all_hyps[algo.op_id]:
                assert algo_param['name'] == search_space.param_names[actual_ind], (
                    algo_param['name'], search_space.param_names[actual_ind])
                param = search_space.params[algo_param['name']]
                if param.is_cont:
                    cont_dims.append(actual_ind)
                    numeric_dims.append(actual_ind)
                elif param.is_disc:
                    disc_dims.append(actual_ind)
                    numeric_dims.append(actual_ind)
                elif param.is_nominal:
                    nominal_dims.append(actual_ind)
                else:
                    raise ValueError(param.dtype)
                actual_ind += 1

            self.params_cont_dims.append(np.array(cont_dims))
            self.params_disc_dims.append(np.array(disc_dims))
            self.params_numeric_dims.append(np.array(numeric_dims))
            self.params_nominal_dims.append(np.array(nominal_dims))


def get_active_dims(seq_len: int, operator_space_id: str, operator_hyperparams_space_id: str):
    # sequence indices
    seq_dims = np.arange(seq_len)
    operator_space = get_operator_space(operator_space_id=operator_space_id)
    operator_hyperparams_space = get_operator_hyperparms_space(
        operator_hyperparams_space_id=operator_hyperparams_space_id)

    # parameter_indices
    total_num_params = 0
    for algo in operator_space.all_operators:
        total_num_params += len(operator_hyperparams_space.all_hyps[algo.op_id])

    param_indices = np.arange(seq_len, seq_len + total_num_params)

    # Get active dimensions for the ConditionalAdditiveKernel
    current_idx = 0
    param_dims = {}  # Dimensions in vector x. Can be used to know which parameters can be changed.
    param_active_dims = {}  # active dims for kernel. Start from 0 for first algorithm

    for algo in operator_space.all_operators:
        n_algo_hyp = len(operator_hyperparams_space.all_hyps[algo.op_id])
        param_dims[algo.op_id] = np.arange(seq_len + current_idx, seq_len + current_idx + n_algo_hyp).tolist()
        param_active_dims[algo.op_id] = np.arange(current_idx, current_idx + n_algo_hyp).tolist()
        current_idx += n_algo_hyp

    return seq_dims, param_indices, param_dims, param_active_dims

import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent)
sys.path[0] = ROOT_PROJECT

from mcbo.search_space.search_space_eda import SearchSpaceEDA
from mcbo.tasks.eda_seq_opt.eda_seq_opt_task import EDASeqOptimization


def test_eda_search_space():
    k = dict(designs_group_id="adder", operator_space_id="new_op_2", seq_operators_pattern_id="free_pattern_52",
             operator_hyperparams_space_id="new_op_2")
    eda = EDASeqOptimization(lut_inputs=6, ref_abc_seq="resyn2", objective="lut", evaluator="abc", **k)
    op_ind_per_type_dic = eda.optim_space.op_ind_per_type_dic
    sp = SearchSpaceEDA(params=eda.optim_space.search_space_params, dtype=torch.float,
                        seq_operators_pattern_id=k["seq_operators_pattern_id"], op_ind_per_type_dic=op_ind_per_type_dic)

    samples = sp.sample(10)

    np_samples_types = np.array(
        [[eda.operator_space.all_operators[i]().op_type.num_id for i in samples.values[l][:sp.seq_len]] for l in
         range(len(samples))])
    str_samples_types = [" ".join(map(str, np_samples_type)) for np_samples_type in np_samples_types]
    assert not np.any(["2 1" in str_samples_type for str_samples_type in str_samples_types])
    assert not np.any(["0 2" in str_samples_type for str_samples_type in str_samples_types])

    transf_samples = sp.transform(samples)
    back_samples = sp.inverse_transform(transf_samples)

    assert np.all(samples.values == back_samples.values)

    for _ in tqdm(range(1000)):
        sample_ind = np.random.randint(0, len(transf_samples))
        seq_ind = np.random.randint(0, sp.seq_len)
        new_values = sp.get_transformed_mutation_cand_values(transf_samples[sample_ind], seq_ind)
        transf_samples[sample_ind, seq_ind] = np.random.choice(new_values).item()

    samples = sp.inverse_transform(transf_samples)
    np_samples_types = np.array(
        [[eda.operator_space.all_operators[i]().op_type.num_id for i in samples.values[l][:sp.seq_len]] for l in
         range(len(samples))])
    str_samples_types = [" ".join(map(str, np_samples_type)) for np_samples_type in np_samples_types]
    assert not np.any(["2 1" in str_samples_type for str_samples_type in str_samples_types])
    assert not np.any(["0 2" in str_samples_type for str_samples_type in str_samples_types])

    return 0


if __name__ == "__main__":
    if test_eda_search_space() == 0:
        print("test_eda_search_space passed")

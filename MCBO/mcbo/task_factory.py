# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from mcbo.tasks import TaskBase, PestControl, CDRH3Design, \
    MigSeqOpt
from mcbo.tasks.synthetic.sfu.sfu_task import SfuTask
from mcbo.tasks.synthetic.sfu.utils_sfu import SFU_FUNCTIONS


def task_factory(task_name: str, **kwargs) -> TaskBase:
    """
    The task name specifies the task that should be returned.
    
    Args:
        task_name: name of the optimization task

    Returns:
        task: the optimization task
    """

    if task_name in SFU_FUNCTIONS:
        task = SfuTask(
            task_name=task_name,
            variable_type=kwargs.pop("variable_type"),
            num_dims=kwargs.pop("num_dims"),
            lb=kwargs.pop("lb", None),
            ub=kwargs.pop("ub", None),
            num_categories=kwargs.pop("num_categories", None),
            **kwargs
        )

    elif task_name == 'pest':
        task = PestControl()

    elif task_name == 'antibody_design':
        task = CDRH3Design(
            antigen=kwargs.get('antigen', '2DD8_S'),
            cdrh3_length=kwargs.get('cdrh3_length', 11),
            num_cpus=kwargs.get('num_cpus', 1),
            first_cpu=kwargs.get('first_cpu', 0),
            absolut_dir=kwargs.get('absolut_dir', None)
        )

    elif task_name == "rna_inverse_fold":
        from mcbo.tasks.rna_inverse_fold.rna_inverse_fold_task import RNAInverseFoldTask
        target = kwargs.get("target", 65)
        if isinstance(target, int):
            from mcbo.tasks.rna_inverse_fold.utils import get_target_from_id
            target = get_target_from_id(target)
        binary_mode = kwargs.get("binary_mode", False)
        task = RNAInverseFoldTask(target=target, binary_mode=binary_mode)

    elif task_name == "xgboost_opt":
        dataset_id = kwargs.get("dataset_id", "mnist")
        split = kwargs.get("split", .3)
        split_seed = kwargs.get("split_seed", 0)
        from mcbo.tasks.xgboost_opt.xgboost_opt_task import XGBoostTask
        task = XGBoostTask(dataset_id=dataset_id, split=split, split_seed=split_seed)

    elif task_name == "svm_opt":
        from mcbo.tasks.svm_opt.svm_opt import SVMOptTask
        task = SVMOptTask()

    elif "aig_optimization" in task_name:
        from mcbo.tasks.eda_seq_opt.eda_seq_opt_task import EDASeqOptimization

        designs_group_id = kwargs.get("designs_group_id", "adder")
        operator_space_id = kwargs.get("operator_space_id", "basic")
        seq_operators_pattern_id = kwargs.get("seq_operators_pattern_id", "basic")
        evaluator = kwargs.get("evaluator", "abc")
        return_best_intermediate = kwargs.get("return_best_intermediate", True)
        lut_inputs = kwargs.get("lut_inputs", "6")
        ref_abc_seq = kwargs.get("ref_abc_seq", "resyn2")
        objective = kwargs.get("objective", "lut")
        n_parallel = kwargs.get("n_parallel", None)
        if task_name == "aig_optimization":
            operator_hyperparams_space_id = None
        elif task_name == "aig_optimization_hyp":
            operator_hyperparams_space_id = kwargs.get("operator_hyperparams_space_id", "boils_hyp_op_space")
        else:
            raise ValueError(task_name)
        task = EDASeqOptimization(
            designs_group_id=designs_group_id, operator_space_id=operator_space_id,
            seq_operators_pattern_id=seq_operators_pattern_id,
            operator_hyperparams_space_id=operator_hyperparams_space_id,
            evaluator=evaluator, lut_inputs=lut_inputs, ref_abc_seq=ref_abc_seq,
            objective=objective, n_parallel=n_parallel,
            return_best_intermediate=return_best_intermediate
        )

    elif task_name == 'mig_optimization':
        seq_len = kwargs.get('seq_len', 20)
        ntk_name = kwargs.get('ntk_name', 'sqrt')
        objective = kwargs.get('objective', 'both')
        task = MigSeqOpt(ntk_name=ntk_name, objective=objective, seq_len=seq_len)

    else:
        raise NotImplementedError(f'Task {task_name} is not implemented.')

    return task

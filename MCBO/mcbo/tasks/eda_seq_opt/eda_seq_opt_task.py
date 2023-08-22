import itertools
import os
from subprocess import CalledProcessError
from typing import List, Tuple, Dict, Any, Type, Optional

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from mcbo.search_space.search_space_eda import SearchSpaceEDA
from mcbo.tasks.eda_seq_opt.utils.utils import get_results_storage_path_root, get_design_name, get_design_prop, \
    get_eda_obj_func, initialise_eda_task
from mcbo.tasks.eda_seq_opt.utils.utils_build_in_seq import get_ref
from mcbo.tasks.eda_seq_opt.utils.utils_design_groups import get_designs_path
from mcbo.tasks.eda_seq_opt.utils.utils_eda_search_space import OptimHyperSpace
from mcbo.tasks.eda_seq_opt.utils.utils_operators import OperatorSpace, get_operator_space, SeqOperatorsPattern, \
    get_seq_operators_pattern, Operator, PRE_MAPPING_OPERATOR_TYPE, \
    make_operator_sequence_valid, is_lut_mapping_hyperparam, MAPPING_OPERATOR_TYPE, POST_MAPPING_OPERATOR_TYPE, If
from mcbo.tasks.eda_seq_opt.utils.utils_operators_hyp import OperatorHypSpace, get_operator_hyperparms_space
from mcbo.tasks.task_base import TaskBase
from mcbo.utils.general_utils import safe_load_w_pickle, log


class EDASeqOptimization(TaskBase):

    @property
    def name(self) -> str:
        name = f'EDA Sequence Optimization - Design {self.designs_group_id} - ' \
               f'Ops {self.operator_space_id} - Pattern {self.seq_operators_pattern_id}'
        if self.operator_hyperparams_space_id is not None and len(self.operator_hyperparams_space_id):
            name += f" - Hyps {self.operator_hyperparams_space_id}"
        name += f" - Obj {self.objective}"
        return name

    def __init__(self, designs_group_id: str, operator_space_id: str, seq_operators_pattern_id: Optional[str],
                 operator_hyperparams_space_id: Optional[str],
                 evaluator: str, lut_inputs: int,
                 ref_abc_seq: str, objective: str, return_best_intermediate: bool, n_parallel: int = 1,
                 n_eval_ref: int = 1, verbose: bool = False):
        """
        Args:
            return_best_intermediate: when evaluating a sequence, whether to return final stats or best inner stats
        """

        self.verbose = verbose

        initialise_eda_task()

        super(EDASeqOptimization, self).__init__()

        assert evaluator in ['abcpy', 'yosys', 'abc'], evaluator
        self.evaluator = evaluator
        self.return_best_intermediate = return_best_intermediate

        self.exec_time = 0
        self.n_parallel = n_parallel

        # number of times ref evaluation should be repeated (important when exexution time matters)
        self.n_eval_ref = n_eval_ref

        self.designs_group_id = designs_group_id
        self.design_files = get_designs_path(self.designs_group_id)
        self.design_names = list(
            map(lambda design_path: os.path.basename(design_path).split('.')[0], self.design_files))

        if ref_abc_seq is None:
            ref_abc_seq = 'init'  # evaluate initial design
        self.ref_abc_seq = ref_abc_seq
        self.lut_inputs = lut_inputs

        self.refs_1: List[float] = []
        self.refs_2: List[float] = []
        self.ref_exec_times: List[float] = []

        self.operator_space_id = operator_space_id
        self.seq_operators_pattern_id = seq_operators_pattern_id
        if operator_hyperparams_space_id:
            self.operator_hyperparams_space_id = operator_hyperparams_space_id
        else:
            self.operator_hyperparams_space_id = ""

        self.objective = objective

        self.operator_space_id = operator_space_id
        self.operator_space: OperatorSpace = self.get_operator_space()
        self.operator_space.check()

        self.hyperparams_space: Optional[OperatorHypSpace] = self.get_operators_hyperparams_space()

        self.seq_operators_pattern: SeqOperatorsPattern = self.get_seq_operators_pattern()
        assert self.seq_operators_pattern is not None
        if self.seq_operators_pattern.not_mapped_at_the_end():
            self.final_mapping_op = If(**{"If K": lut_inputs})
        else:
            self.final_mapping_op = None

        self.optim_space: OptimHyperSpace = OptimHyperSpace(
            operator_space=self.operator_space,
            seq_operators_pattern=self.seq_operators_pattern,
            operator_hyperparams_space=self.hyperparams_space
        )

        self.ckpt_data: EDAOptSeqCkpt = EDAOptSeqCkpt()

        self.ckpt_data.build_data(search_space_len=self.optim_space.search_space_len,
                                  n_designs=len(self.design_files),
                                  n_intermediate_vals=self.n_print_stats,
                                  n_out_dims=self.n_out_dims)

    @property
    def n_out_dims(self) -> int:
        if self.objective in ["exec_time_under_constr_1"]:
            return 3
        else:
            return 1

    @property
    def n_print_stats(self) -> int:
        return self.seq_operators_pattern.n_print_stats + (self.final_mapping_op is not None)

    def get_eda_obj_func(self):
        return get_eda_obj_func(self.objective)

    def get_ckpt_data(self) -> Dict[str, Any]:
        return self.ckpt_data.get_ckpt_data_as_dict()

    def load_ckpt_data(self, ckpt: Dict[str, Any]):
        self.ckpt_data.load_ckpt_data(ckpt, n_intermediate_vals=self.n_print_stats)
        self._n_bb_evals = len(self.ckpt_data.samples_X)

    def get_ref(self, design_file: str) -> Tuple[float, float, float]:
        """ Return either area and delay or lut and levels obtained when applying a given sequence on a given
        design file.
        """
        return get_ref(
            design_file=design_file,
            lut_inputs=self.lut_inputs,
            ref_abc_seq=self.ref_abc_seq,
            evaluator=self.evaluator,
            n_eval_ref=self.n_eval_ref
        )

    def get_config(self) -> Dict[str, Any]:
        return dict(
            design_files_id=self.designs_group_id,
            seq_operators_pattern_id=self.seq_operators_pattern_id,
            operator_space_id=self.operator_space_id,
            operator_hyperparams_space_id=self.operator_hyperparams_space_id,
            lut_inputs=self.lut_inputs,
            ref_abc_seq=self.ref_abc_seq,
            evaluator=self.evaluator,
            objective=self.objective,
            n_eval_ref=self.n_eval_ref,
            return_best_intermediate=self.return_best_intermediate,
        )

    def exp_path(self) -> str:
        return self.get_exp_path_aux(
            design_files_id=self.designs_group_id,
            seq_operators_pattern_id=self.seq_operators_pattern_id,
            operator_space_id=self.operator_space_id,
            operator_hyperparams_space_id=self.operator_hyperparams_space_id,
            lut_inputs=self.lut_inputs,
            ref_abc_seq=self.ref_abc_seq,
            evaluator=self.evaluator,
            objective=self.objective,
            n_eval_ref=self.n_eval_ref,
            return_best_intermediate=self.return_best_intermediate,
        )

    @staticmethod
    def get_exp_path_aux(evaluator: str, objective: str,
                         lut_inputs: int,
                         seq_operators_pattern_id: str,
                         operator_space_id: str,
                         operator_hyperparams_space_id: str,
                         design_files_id: str, ref_abc_seq: str,
                         n_eval_ref: int, return_best_intermediate: bool) -> str:
        exp_id = f"lut-{lut_inputs}" \
                 f"_eval-{evaluator}" \
                 f"_obj-{objective}" \
                 f"_seq-{seq_operators_pattern_id if seq_operators_pattern_id else ''}" \
                 f"_ref-{ref_abc_seq}" \
                 f"_op-{operator_space_id}" \
                 f"_hyp-{operator_hyperparams_space_id}"
        if n_eval_ref != 1:
            exp_id += f"_n-ref-eval-{n_eval_ref}"
        if return_best_intermediate:
            exp_id += "_best-inter"
        return os.path.join(get_results_storage_path_root(),
                            exp_id,
                            design_files_id)

    @staticmethod
    def get_eval_ckpt_root_path(
            lut_inputs: int, evaluator: str, operator_space_id: str,
            seq_operators_pattern_id: str, operator_hyperparams_space_id: str,
            return_best_intermediate: bool
    ) -> str:
        eval_ckpt_id = f"lut-{lut_inputs}" \
                       f"_evaluator-{evaluator}" \
                       f"_op-{operator_space_id}" \
                       f"_seq-{seq_operators_pattern_id if seq_operators_pattern_id else ''}" \
                       f"_hyp-{operator_hyperparams_space_id}"
        if return_best_intermediate:
            eval_ckpt_id += "_best-inter"
        return os.path.join(
            get_results_storage_path_root(),
            eval_ckpt_id
        )

    @property
    def eval_ckpt_root_path(self) -> str:
        return self.get_eval_ckpt_root_path(
            lut_inputs=self.lut_inputs,
            evaluator=self.evaluator,
            operator_space_id=self.operator_space_id,
            seq_operators_pattern_id=self.seq_operators_pattern_id,
            operator_hyperparams_space_id=self.operator_hyperparams_space_id,
            return_best_intermediate=self.return_best_intermediate
        )

    @property
    def operator_space_length(self) -> int:
        return len(self.operator_space)

    def get_operator_space(self):
        return get_operator_space(self.operator_space_id)

    def get_seq_operators_pattern(self) -> Optional[SeqOperatorsPattern]:
        return get_seq_operators_pattern(self.seq_operators_pattern_id)

    def get_operators_hyperparams_space(self) -> Optional[OperatorHypSpace]:
        return get_operator_hyperparms_space(self.operator_hyperparams_space_id)

    @staticmethod
    def convert_array_to_operator_seq(sequence: np.ndarray, hyperparams: Dict[str, Any], operator_space: OperatorSpace,
                                      lut_inputs: int,
                                      final_mapping_op: Optional[Operator]) -> Tuple[List[Operator], np.ndarray, bool]:
        assert sequence.ndim == 1, sequence.shape
        operator_sequence: List[Operator] = []
        print_stat_stages = [0]
        for ind in sequence:
            assert ind == np.round(ind), ind
            ind = int(np.round(ind))
            op_class: Type[Operator] = operator_space.all_operators[ind]
            op_hyp: Dict[str, Any] = {}
            for kw in op_class.hyperparams:
                if kw in hyperparams:
                    op_hyp[kw] = hyperparams[kw]
                if is_lut_mapping_hyperparam(kw):
                    op_hyp[kw] = lut_inputs
            operator_sequence.append(op_class(**op_hyp))
            if (operator_sequence[-1].op_type in [MAPPING_OPERATOR_TYPE,
                                                  POST_MAPPING_OPERATOR_TYPE]) or final_mapping_op:
                print_stat_stages.append(print_stat_stages[-1] + 1)  # new print_stats
            else:
                print_stat_stages.append(print_stat_stages[-1])  # no new print_stats
        assert operator_sequence[0].op_type == PRE_MAPPING_OPERATOR_TYPE, operator_sequence[0]
        valid_seq_operator, is_new_op = make_operator_sequence_valid(operator_sequence,
                                                                     final_mapping_op=final_mapping_op)
        return valid_seq_operator, np.array(print_stat_stages), is_new_op

    def evaluate_(self, sequence: np.ndarray, hyperparams: Dict[str, Any],
                  design_file: str, ref_1: float, ref_2: float, ref_exec_time: float,
                  n_evals: int) -> Tuple[float, float, bool, Dict[str, Any]]:
        """ Return either area and delay or lut and levels """

        # design_id = get_design_name(design_file)

        # eval_dic_path = os.path.join(self.eval_ckpt_root_path, design_id, 'eval.pkl')
        # if not os.path.exists(eval_dic_path):
        #     os.makedirs(os.path.dirname(eval_dic_path), exist_ok=True)
        #     save_w_pickle({}, eval_dic_path)
        eval_dic = {}  # safe_load_w_pickle(eval_dic_path)

        operator_sequence, print_stat_stages, is_new_op = self.convert_array_to_operator_seq(
            sequence=sequence,
            hyperparams=hyperparams,
            operator_space=self.operator_space,
            lut_inputs=self.lut_inputs,
            final_mapping_op=self.final_mapping_op
        )
        if not self.seq_operators_pattern.contains_free:
            print_stat_stages = None

        seq_ind_id = ";".join(map(lambda op: op.op_str(), operator_sequence))
        if is_new_op:
            seq_ind_id = f"&get -n -m; {seq_ind_id}"
        if self.evaluator == "abcpy":
            raise ValueError("Needs to add strash after post-mapping operations and add rec_start3 call for LMS")
        sequence = [(operator.op_id if self.evaluator == 'abcpy' else
                     operator.op_str()) for operator in operator_sequence]
        if is_new_op:
            sequence = ["&get -n -m;"] + sequence
        if seq_ind_id in eval_dic and len(eval_dic[seq_ind_id]) >= 3 and "luts" in eval_dic[seq_ind_id][2] \
                and len(eval_dic[seq_ind_id][2]["luts"]) == self.n_print_stats \
                and eval_dic[seq_ind_id][2] is not None \
                and "exec_time" in eval_dic[seq_ind_id][2] \
                and eval_dic[seq_ind_id][2]["exec_time"] > 0:
            if n_evals % 50 == 1 and self.verbose:
                self.log(f"{n_evals}. Already computed {seq_ind_id} for {get_design_name(design_file)}...")
            obj_1, obj_2, extra_info, valid = eval_dic[seq_ind_id]
            exec_time = extra_info["exec_time"]
        else:
            valid = True
            try:
                if n_evals % 50 == 1 and self.verbose:
                    log(f"{n_evals}. Evaluate {seq_ind_id}",
                        header=f"{self.objective}. -- {get_design_name(design_file)}")
                obj_func = None
                if self.return_best_intermediate:
                    aux_obj_func = self.get_eda_obj_func()

                    def custom_obj_func(val_1, val_2) -> float:
                        return aux_obj_func(val_1 / ref_1, val_2 / ref_2, ref_1=ref_1, ref_2=ref_2)

                    obj_func = custom_obj_func
                obj_1, obj_2, extra_info = get_design_prop(
                    seq=sequence,
                    design_file=design_file,
                    evaluator=self.evaluator,
                    print_stat_stages=print_stat_stages,
                    new_op=is_new_op,
                    obj_func=obj_func
                )
                exec_time = extra_info["exec_time"]
            except CalledProcessError as e:
                if e.args[0] == -6:
                    log(f"Got error with design: {get_design_name(design_file)} -> setting objs to refs ")
                    obj_1 = ref_1
                    obj_2 = ref_2
                    valid = False
                    extra_info = None
                    exec_time = 5 * ref_exec_time  # penalise invalid sequence
                else:
                    raise e
            # eval_dic = safe_load_w_pickle(eval_dic_path, n_trials=5, time_sleep=2 + np.random.random() * 3)
            # eval_dic[seq_ind_id] = obj_1, obj_2, extra_info, valid
            # save_w_pickle(eval_dic, eval_dic_path)

        if extra_info is None:
            extra_info = {}
        extra_keys = ["luts", "levels", "edges"]
        for extra_key in extra_keys:
            if extra_key not in extra_info:
                extra_info[extra_key] = [0 for _ in range(self.n_print_stats)]
        if "exec_time" not in extra_info:
            extra_info["exec_time"] = exec_time
        return obj_1 / ref_1, obj_2 / ref_2, valid, extra_info

    def df_to_seq_hyp(self, x: pd.Series) -> Tuple[np.ndarray, Dict[str, Any]]:
        x_seq = x.values[:self.optim_space.seq_len]
        x_hyp = {k: x[k] for k in x.index[self.optim_space.seq_len:]}
        return x_seq, x_hyp

    def compute_refs(self):
        refs = Parallel(n_jobs=self.n_parallel, backend="multiprocessing")(
            delayed(self.get_ref)(self.design_files[ind])
            for ind in tqdm(range(len(self.design_files))))

        for refs_1_2 in refs:
            self.refs_1.append(refs_1_2[0])
            self.refs_2.append(refs_1_2[1])
            self.ref_exec_times.append(refs_1_2[2])

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        if len(self.refs_1) == 0:
            self.compute_refs()

        n = len(x)

        ind_prod = list(
            itertools.product(range(0, n), range(len(self.design_files))))

        inputs = [self.df_to_seq_hyp(x_i[1]) for x_i in x.iterrows()]
        x_seqs = [x_full[0] for x_full in inputs]
        x_hyps = [x_full[1] for x_full in inputs]

        objs = Parallel(n_jobs=self.n_parallel, backend="multiprocessing")(
            delayed(self.evaluate_)(sequence=x_seqs[ind_sample],
                                    hyperparams=x_hyps[ind_sample],
                                    design_file=self.design_files[ind_design],
                                    ref_1=self.refs_1[ind_design],
                                    ref_2=self.refs_2[ind_design],
                                    ref_exec_time=self.ref_exec_times[ind_design],
                                    n_evals=self._n_bb_evals)
            for ind_sample, ind_design in ind_prod
        )

        new_objs_1 = np.zeros((n, len(self.design_files)))
        new_objs_2 = np.zeros((n, len(self.design_files)))
        new_ex_times = np.zeros((n, len(self.design_files)))
        new_intermediate_objs_1 = np.zeros((n, len(self.design_files), self.n_print_stats))
        new_intermediate_objs_2 = np.zeros((n, len(self.design_files), self.n_print_stats))
        new_intermediate_edges = np.zeros((n, len(self.design_files), self.n_print_stats))
        new_full_valids = np.zeros((n, len(self.design_files)))

        obj_index = 0
        for ind_sample_ in range(n):
            for ind_design_ in range(len(self.design_files)):
                new_objs_1[ind_sample_, ind_design_] = objs[obj_index][0]
                new_objs_2[ind_sample_, ind_design_] = objs[obj_index][1]
                new_full_valids[ind_sample_, ind_design_] = objs[obj_index][2]
                new_ex_times[ind_sample_, ind_design_] = objs[obj_index][3]["exec_time"]
                new_intermediate_objs_1[ind_sample_, ind_design_] = np.array(objs[obj_index][3]["luts"])
                new_intermediate_objs_2[ind_sample_, ind_design_] = np.array(objs[obj_index][3]["levels"])
                new_intermediate_edges[ind_sample_, ind_design_] = np.array(objs[obj_index][3]["edges"])
                obj_index += 1

        # Compute objective value for all designs and input sequences
        if self.objective == "exec_time_under_constr_1":
            full_fX = np.array([
                [new_ex_times[ind_sample, ind_design] / self.ref_exec_times[ind_design] for
                 ind_design in range(len(self.design_files))] for ind_sample in range(n)]
            )
            constr_vals_1 = new_objs_1.max(1)
            constr_vals_2 = new_objs_2.max(1)
            fX = np.vstack([np.mean(full_fX, axis=1), constr_vals_1, constr_vals_2]).T
            assert fX.shape == (n, 3), (fX.shape, (n, 3))
        else:
            obj_func = self.get_eda_obj_func()
            full_fX = np.array(
                [[obj_func(new_objs_1[ind_sample, ind_design], new_objs_2[ind_sample, ind_design],
                           ref_1=self.refs_1[ind_design], ref_2=self.refs_2[ind_design]) for
                  ind_design in range(len(self.design_files))] for ind_sample in range(n)])
            assert full_fX.shape == (n, len(self.design_files)), full_fX.shape

            fX = np.mean(full_fX, axis=1)
            fX = np.array(fX, dtype=np.float64).reshape((n, 1))

        self.ckpt_data.update(
            new_samples_X=x.values, new_ys=fX, new_full_objs_1_s=new_objs_1,
            new_full_objs_2_s=new_objs_2,
            new_full_valids=new_full_valids,
            new_intermediate_objs_1=new_intermediate_objs_1,
            new_intermediate_objs_2=new_intermediate_objs_2,
            new_execution_times=new_ex_times,
            new_intermediate_edges=new_intermediate_edges
        )
        return fX

    @property
    def obj1_id(self):
        return f'lut_{self.lut_inputs}'

    @property
    def obj2_id(self):
        return f'levels'

    def log(self, msg: str, end=None) -> None:
        log(msg, header=f"{self.designs_group_id} | {self.objective}", end=end)

    def get_search_space_params(self):
        return self.optim_space.search_space_params

    def get_search_space(self, dtype: torch.dtype = torch.float64) -> SearchSpaceEDA:
        return SearchSpaceEDA(
            self.optim_space.search_space_params, dtype=dtype,
            seq_operators_pattern_id=self.seq_operators_pattern_id,
            op_ind_per_type_dic=self.optim_space.op_ind_per_type_dic
        )


class EDAOptSeqCkpt:

    def __init__(self):
        self.samples_X = None
        self.ys = None
        self.full_obj_1_s = None
        self.full_obj_2_s = None
        self.full_valids = None
        self.full_execution_times = None
        self.intermediate_objs_1 = None
        self.intermediate_objs_2 = None
        self.intermediate_edges = None

    def load_ckpt_data_from_path(self, path: str, n_intermediate_vals: Optional[int] = None) -> None:
        ckpt = safe_load_w_pickle(path)
        self.load_ckpt_data(ckpt=ckpt, n_intermediate_vals=n_intermediate_vals)

    def load_ckpt_data(self, ckpt: Dict[str, Any], n_intermediate_vals: Optional[int] = None) -> None:
        self.samples_X = ckpt['X']
        self.ys = ckpt['y']
        self.full_obj_1_s = ckpt['full_objs_1']
        self.full_obj_2_s = ckpt['full_objs_2']
        self.full_valids = ckpt['full_valids']
        extra_keys = ["intermediate_objs_1", "intermediate_objs_2", "intermediate_edges"]
        for extra_key in extra_keys:
            if extra_key not in ckpt:
                assert n_intermediate_vals is not None
                setattr(self, extra_key, np.zeros((*self.full_obj_1_s.shape, n_intermediate_vals)))
            else:
                setattr(self, extra_key, ckpt[extra_key])
        simple_extra_keys = ["full_execution_times"]
        for extra_key in simple_extra_keys:
            if extra_key not in ckpt:
                assert n_intermediate_vals is not None
                setattr(self, extra_key, np.zeros(self.full_obj_1_s.shape))
            else:
                setattr(self, extra_key, ckpt[extra_key])

    def get_ckpt_data_as_dict(self) -> Dict[str, Any]:
        assert self.samples_X is not None
        return {
            'X': self.samples_X,
            'y': self.ys,
            'full_objs_1': self.full_obj_1_s,
            'full_objs_2': self.full_obj_2_s,
            'full_valids': self.full_valids,
            'full_execution_times': self.full_execution_times,
            "intermediate_objs_1": self.intermediate_objs_1,
            "intermediate_objs_2": self.intermediate_objs_2,
            "intermediate_edges": self.intermediate_edges
        }

    def build_data(self, search_space_len: int, n_designs: int, n_out_dims: int, n_intermediate_vals: int) -> None:
        self.samples_X = np.zeros((0, search_space_len))
        self.ys = np.zeros((0, n_out_dims))
        self.full_obj_1_s = np.zeros((0, n_designs))
        self.full_obj_2_s = np.zeros((0, n_designs))
        self.full_valids = np.zeros((0, n_designs))
        self.full_execution_times = np.zeros((0, n_designs))
        self.intermediate_objs_1 = np.zeros((0, n_designs, n_intermediate_vals))
        self.intermediate_objs_2 = np.zeros((0, n_designs, n_intermediate_vals))
        self.intermediate_edges = np.zeros((0, n_designs, n_intermediate_vals))

    def update(self, new_samples_X: np.ndarray, new_ys: np.ndarray, new_full_objs_1_s: np.ndarray,
               new_full_objs_2_s: np.ndarray, new_full_valids: np.ndarray, new_execution_times: np.ndarray,
               new_intermediate_objs_1: np.ndarray, new_intermediate_objs_2: np.ndarray,
               new_intermediate_edges: np.ndarray):
        self.samples_X = np.vstack([self.samples_X, new_samples_X])
        self.ys = np.vstack([self.ys, new_ys])
        self.full_obj_1_s = np.vstack([self.full_obj_1_s, new_full_objs_1_s])
        self.full_obj_2_s = np.vstack([self.full_obj_2_s, new_full_objs_2_s])
        self.full_valids = np.vstack([self.full_valids, new_full_valids])
        self.full_execution_times = np.vstack([self.full_execution_times, new_execution_times])
        self.intermediate_objs_1 = np.vstack([self.intermediate_objs_1, new_intermediate_objs_1])
        self.intermediate_objs_2 = np.vstack([self.intermediate_objs_2, new_intermediate_objs_2])
        self.intermediate_edges = np.vstack([self.intermediate_edges, new_intermediate_edges])


def np_to_seq_hyp(x: np.ndarray, optim_space: OptimHyperSpace) -> Tuple[np.ndarray, Dict[str, Any]]:
    assert x.ndim == 1, x.shape
    x_seq = x[:optim_space.seq_len]
    x_hyp = {optim_space.search_space_params[i]['name']: x[i] for i in range(optim_space.seq_len, len(x))}
    return x_seq, x_hyp


def get_seq_from_np(x: np.ndarray, optim_space: OptimHyperSpace, evaluator: str, lut_inputs: int) -> str:
    sequence, hyperparams = np_to_seq_hyp(x, optim_space=optim_space)

    operator_sequence: List[Operator] = []
    for ind in sequence:
        assert ind == np.round(ind), ind
        ind = int(np.round(ind))
        op_class: Type[Operator] = optim_space.operator_space.all_operators[ind]
        op_hyp: Dict[str, Any] = {}
        for kw in op_class.hyperparams:
            if kw in hyperparams:
                op_hyp[kw] = hyperparams[kw]
            if is_lut_mapping_hyperparam(kw):
                op_hyp[kw] = lut_inputs
        operator_sequence.append(op_class(**op_hyp))
    assert operator_sequence[0].op_type == PRE_MAPPING_OPERATOR_TYPE, operator_sequence[0]
    operator_sequence, new_op = make_operator_sequence_valid(operator_sequence)
    if evaluator == "abcpy":
        raise ValueError("Needs to add strash after post-mapping operations and add rec_start3 call for LMS")
    sequence = [(operator.op_id if evaluator == 'abcpy' else
                 operator.op_str()) for operator in operator_sequence]
    return ";".join(sequence).replace(";;", ";")

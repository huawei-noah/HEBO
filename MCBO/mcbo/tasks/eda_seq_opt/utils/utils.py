import os
import re
import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Callable, Optional

import numpy as np

from mcbo.utils.general_utils import log


def get_abc_release_path() -> str:
    path_to_eda_seq_opt = str(Path(os.path.realpath(__file__)).parent.parent)
    aux_path = os.path.join(path_to_eda_seq_opt, 'abc_release_path.txt')
    try:
        f = open(aux_path, "r")
        abc_release_path = f.readlines()[0]
        if abc_release_path[-1] == '\n':
            abc_release_path = abc_release_path[:-1]
        f.close()
        return abc_release_path
    except FileNotFoundError as _:
        abs_path_to_abc = '/my/absolute/path/to/abc'
        error_message = f'------ Friendly first run error message ------\n' \
                        f'File \n - {aux_path}\nnot found,\n' \
                        f'   --> shall create it and fill it with one line describing the ' \
                        f'path to abc executable (if you want to evaluate sequences using this executable, and not ' \
                        f'using yosys-abc or abcpy, in which case you can just put any path), e.g. running\n' \
                        f"\techo '{abs_path_to_abc}' > {aux_path}\n"
        raise FileNotFoundError(error_message)


def get_circuits_path_root() -> str:
    path_to_eda_seq_opt = str(Path(os.path.realpath(__file__)).parent.parent)
    aux_path = os.path.join(path_to_eda_seq_opt, 'circuits_path.txt')
    try:
        f = open(aux_path, "r")
        abc_release_path = f.readlines()[0]
        if abc_release_path[-1] == '\n':
            abc_release_path = abc_release_path[:-1]
        f.close()
        return abc_release_path
    except FileNotFoundError as _:
        abs_path_to_circuits = '/my/absolute/path/to/blif_circuits/'
        error_message = f'------ Friendly first run error message ------\n' \
                        f'File \n - {aux_path}\nnot found,\n' \
                        f'   --> shall create it and fill it with one line describing the ' \
                        f'path to the .blif circuits, e.g. running\n' \
                        f"\techo '{abs_path_to_circuits}' > {aux_path}\n"
        raise FileNotFoundError(error_message)


def get_results_storage_path_root() -> str:
    path_to_results_root = str(Path(os.path.realpath(__file__)).parent.parent)
    aux_path = os.path.join(path_to_results_root, 'results_root_path.txt')
    try:
        f = open(aux_path, "r")
        abc_release_path = f.readlines()[0]
        if abc_release_path[-1] == '\n':
            abc_release_path = abc_release_path[:-1]
        f.close()
        return abc_release_path
    except FileNotFoundError as _:
        abs_path_to_results_root = '/my/absolute/path/to/root_results/'
        error_message = f'------ Friendly first run error message ------\n' \
                        f'File \n - {aux_path}\nnot found,\n' \
                        f'   --> shall create it and fill it with one line describing the ' \
                        f'path to where you want all the results to be stored, e.g. running\n' \
                        f"\techo '{abs_path_to_results_root}' > {aux_path}\n"
        raise FileNotFoundError(error_message)


def initialise_eda_task():
    for init_funcs in [get_abc_release_path, get_circuits_path_root, get_results_storage_path_root]:
        while 1:
            try:
                init_funcs()
                break
            except FileNotFoundError as e:
                print(e.args[0])
                print(
                    "\n:/ ---------- Solve error following the FileNotFoundError instructions"
                    " and press ENTER to continue ------------- \\:\n")
                input()


def evaluate_cmd(evaluator: str, abc_command: str, print_stat_stages: Optional[np.ndarray], new_op: bool,
                 obj_func: Optional[Callable[[float, float], float]] = None,
                 verbose: int = 0):
    """ Evaluate the sequence of operations described in abc_command using provided `evaluator`

    Args:
        evaluator: @@
        abc_command: command to execute
        verbose: verbosity level
    """
    assert evaluator in ['yosys', 'abc']
    t_ref = time.time()
    extra_info = {}
    if evaluator == 'yosys':
        cmd_elements_op = 'yosys-abc'
    elif evaluator == 'abc':
        cmd_elements_op = get_abc_release_path()
    else:
        raise ValueError(f"{evaluator} not supported.")
    if verbose:
        log(abc_command, header="Sequence evaluation")

    if new_op:
        n_print_stats = abc_command.count("&ps")
    else:
        n_print_stats = abc_command.count("print_stats")

    cmd_elements = [cmd_elements_op, '-c', abc_command]
    proc = subprocess.check_output(cmd_elements)
    # read results and extract information
    levels = []
    luts = []
    objs = []
    edges = []
    if new_op:
        for line in proc.decode("utf-8").split('\n'):
            ob = re.search(r"Mapping \(K=[0-9]\) *: *.*lut *= *[0-9]+.* *.*edge *= *[0-9]+.* *.*lev *= *[0-9]+", line)
            if ob is not None:
                ob = re.search(r'lev *= *[0-9]+', line)
                levels.append(int(ob.group().split('=')[1].strip()))

                ob = re.search(r'lut *= *[0-9]+', line)
                luts.append(int(ob.group().split('=')[1].strip()))

                ob = re.search(r'edge *= *[0-9]+', line)
                edges.append(int(ob.group().split('=')[1].strip()))

                if obj_func is not None:
                    objs.append(obj_func(luts[-1], levels[-1]))
    else:
        for line in proc.decode("utf-8").split('\n'):
            ob = re.search(r"i/o *= *[0-9]+/ *[0-9]+.*nd *= *[0-9]+ *edge *= *[0-9]+.*lev *= *[0-9]+", line)
            if ob is not None:
                ob = re.search(r'lev *= *[0-9]+', line)
                levels.append(int(ob.group().split('=')[1].strip()))

                ob = re.search(r'nd *= *[0-9]+', line)
                luts.append(int(ob.group().split('=')[1].strip()))

                ob = re.search(r'edge *= *[0-9]+', line)
                edges.append(int(ob.group().split('=')[1].strip()))

                if obj_func is not None:
                    objs.append(obj_func(luts[-1], levels[-1]))

    if len(levels) != n_print_stats:
        print("----" * 10)
        print(f'Command: {" ".join(cmd_elements)}')
        line = proc.decode("utf-8").split('\n')[-2].split(':')[-1].strip()
        print(f"Out line: {line}")
        print("----" * 10)
        raise ValueError(f"Problem when executing the command\n"
                         f"{cmd_elements_op} -c \"{abc_command}\"\n"
                         f"\n--> expected stats to be showed {n_print_stats} times, "
                         f"got {len(levels)} times")

    if print_stat_stages is not None:
        assert print_stat_stages[-1] == (n_print_stats - 1), (print_stat_stages, n_print_stats)
        luts = [luts[i] for i in print_stat_stages]
        levels = [levels[i] for i in print_stat_stages]
        edges = [edges[i] for i in print_stat_stages]
        if len(objs) > 0:
            objs = [objs[i] for i in print_stat_stages]

    extra_info['exec_time'] = time.time() - t_ref
    extra_info['luts'] = luts
    extra_info['levels'] = levels
    extra_info['edges'] = edges

    if len(objs) > 0:
        assert len(objs) == len(luts) == len(edges), (len(objs), len(luts))
        best_obj_ind = np.argmin(objs)
        returned_lut = luts[best_obj_ind]
        returned_level = levels[best_obj_ind]
    else:
        returned_lut, returned_level = luts[-1], levels[-1]
    return returned_lut, returned_level, extra_info


def get_design_name(design_filepath: str) -> str:
    return os.path.basename(design_filepath).split('.')[0]


def get_design_prop(seq: List[str], design_file: str, evaluator: str, print_stat_stages: Optional[np.ndarray],
                    new_op: bool, sweep: bool = False,
                    obj_func: Optional[Callable[[float, float], float]] = None, verbose: bool = False) \
        -> Tuple[int, int, Dict[str, Any]]:
    """
     Get property of the design after applying sequence of operations

    Args:
        seq: sequence of operations
        design_file: path to the design
        evaluator: whether to use yosys-abc (`yosys`) or abc_py (`abcpy`) or compiled abc repo (`abc`)
        sweep: add sweep before sequence
        verbose: verbosity level
        obj_func: a function taking lut and level as input and outputting some value (target in a minimisation task)

    Returns:
        either:
            - for fpga: lut_k, level, extra_info
            - for scl: area, delay
    """
    if seq is None:
        seq = ['strash; ']
        if new_op:
            seq += ['&get -n -m;']

    abc_command = 'read ' + design_file + '; '
    if sweep:
        abc_command += "sweep; sweep -s;"
    abc_command += ' '.join(seq)
    if new_op:
        if "&ps" not in abc_command[-19:]:
            abc_command += '&ps; '
    else:
        if "print_stats" not in abc_command[-19:]:
            abc_command += 'print_stats; '

    lut_k, levels, extra_info = evaluate_cmd(evaluator=evaluator, abc_command=abc_command,
                                             print_stat_stages=print_stat_stages, new_op=new_op, obj_func=obj_func,
                                             verbose=verbose)
    return lut_k, levels, extra_info


# --- Standard objective functions when aggregating two outputs (such as LUT-count and Levels)
def obj_both(ratio_1: float, ratio_2: float, **kwargs) -> float:
    return ratio_1 + ratio_2


def obj_level(ratio_1: float, ratio_2: float, **kwargs) -> float:
    return ratio_2


def obj_lut(ratio_1: float, ratio_2: float, **kwargs) -> float:
    return ratio_1


def obj_lut_sig_level(ratio_1: float, ratio_2: float, **kwargs) -> float:
    """ Return Val_1 / Ref_1 + 2 * (sigmoid(Val_2) - .5) / Ref_1 --> main focus is on Val_1 optimization """
    ref_1 = kwargs["ref_1"]
    return ratio_1 + pseudo_sig(ratio_2) / ref_1


def obj_level_sig_lut(ratio_1: float, ratio_2: float, **kwargs) -> float:
    """ Return Val_2 / Ref_2 + 2 * (sigmoid(Val_1) - .5)  / Ref_2 --> main focus is on Val_2 optimization """
    ref_2 = kwargs["ref_2"]
    return ratio_2 + pseudo_sig(ratio_1) / ref_2


def obj_min_improvements(ratio_1: float, ratio_2: float, **kwargs) -> float:
    """ improvement is 1 - ratio so to maximise the minimal improvement we need to minimise the maximal ratio """
    return max(ratio_1, ratio_2)


def obj_level_marg_lut(ratio_1, ratio_2, **kwargs):
    return ratio_2 + (1 / (1 + 1 / ratio_1))


def obj_lut_marg_level(ratio_1, ratio_2, **kwargs):
    return ratio_1 + (1 / (1 + 1 / ratio_2))


def get_obj_both_with_constr(lut_ratio_constr: Optional[float] = None, level_ratio_constr: Optional[float] = None,
                             **kwargs) -> \
        Callable[[float, float], float]:
    def obj(lut_ratio: float, level_ratio: float) -> float:
        if lut_ratio_constr and lut_ratio > lut_ratio_constr:  # constraint is not fulfilled
            # set penalty
            if level_ratio_constr:
                level_ratio = max(level_ratio, level_ratio_constr)
            else:
                level_ratio = max(lut_ratio_constr, level_ratio)
        elif level_ratio_constr and level_ratio > level_ratio_constr:  # constraint is not fulfilled
            # set penalty
            if lut_ratio_constr:
                lut_ratio = max(lut_ratio, lut_ratio_constr)
            else:
                lut_ratio = max(lut_ratio, level_ratio_constr)
        return lut_ratio + level_ratio

    return obj


def obj_both_with_constr_1p1():
    return get_obj_both_with_constr(lut_ratio_constr=1.1, level_ratio_constr=1.1)


def obj_lut_with_level_constr(ratio_1: float, ratio_2: float):
    return ratio_1 + ratio_2


EDA_OBJ_FUNCS: Dict[str, Callable[[float, float], float]] = {
    "both": obj_both,
    "level": obj_level,
    "lut": obj_lut,
    "lut_psig_level": obj_lut_sig_level,
    "level_psig_lut": obj_level_sig_lut,
    "lut_marg_level": obj_lut_marg_level,
    "level_marg_lut": obj_level_marg_lut,
    "min_improvement": obj_min_improvements,
    "both_constr_1p1": obj_both_with_constr_1p1()
}


def get_eda_obj_func(eda_obj_id: str) -> Callable:
    return EDA_OBJ_FUNCS[eda_obj_id]


def get_eda_available_obf_funcs() -> List[str]:
    return list(EDA_OBJ_FUNCS.keys())


def pseudo_sig(x):
    return 2 * (1 / (1 + np.exp(-x)) - .5)


# ------------------------------------------------------------------------------------------------

class EDAExpPathManager:

    def __init__(self, task_root_path: str, opt_id: str, seed: int):
        self.opt_id = opt_id
        self.task_root_path = task_root_path
        self.seed = seed

    def eda_seq_opt_result_path_root(self):
        return self.get_eda_seq_opt_result_path_root(
            task_root_path=self.task_root_path,
            opt_id=self.opt_id,
            seed=self.seed
        )

    @staticmethod
    def get_eda_seq_opt_result_path_root(task_root_path: str, opt_id: str, seed: int) -> str:
        return os.path.join(task_root_path, opt_id, f"seed-{seed}")

    def eda_seq_opt_result_full_ckpt_path(self):
        return self.get_eda_seq_opt_result_full_ckpt_path(
            task_root_path=self.task_root_path,
            opt_id=self.opt_id,
            seed=self.seed
        )

    @staticmethod
    def get_eda_seq_opt_result_full_ckpt_path(task_root_path: str, opt_id: str, seed: int) -> str:
        return os.path.join(
            EDAExpPathManager.get_eda_seq_opt_result_path_root(
                task_root_path=task_root_path,
                opt_id=opt_id,
                seed=seed
            ), "full_ckpt.pkl")

    def eda_seq_opt_is_running_path(self):
        return self.get_eda_seq_opt_is_running_path(
            task_root_path=self.task_root_path,
            opt_id=self.opt_id,
            seed=self.seed
        )

    @staticmethod
    def get_eda_seq_opt_is_running_path(task_root_path: str, opt_id: str, seed: int) -> str:
        return os.path.join(
            EDAExpPathManager.get_eda_seq_opt_result_path_root(
                task_root_path=task_root_path,
                opt_id=opt_id,
                seed=seed
            ), "is_running.pkl")

    def eda_seq_opt_optimizer_path(self):
        return self.get_eda_seq_opt_optimizer_path(
            task_root_path=self.task_root_path,
            opt_id=self.opt_id,
            seed=self.seed
        )

    @staticmethod
    def get_eda_seq_opt_optimizer_path(task_root_path: str, opt_id: str, seed: int) -> str:
        return os.path.join(
            EDAExpPathManager.get_eda_seq_opt_result_path_root(
                task_root_path=task_root_path,
                opt_id=opt_id,
                seed=seed
            ), "optimizer.pkl")

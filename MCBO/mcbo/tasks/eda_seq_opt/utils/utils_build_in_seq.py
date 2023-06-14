# Comes from BOiLS

import abc
import os
from typing import List, Tuple, Dict, Union, Type, Any
import numpy as np
from mcbo.tasks.eda_seq_opt.utils.utils import get_design_prop, get_results_storage_path_root
from mcbo.utils.general_utils import load_w_pickle, save_w_pickle


class BuildInSeq(abc.ABC):
    sequence: List[str]

    def __init__(self, design_file: str):
        self.design_file = design_file

    def fpga(self, evaluator: str, verbose: bool = False) \
            -> Tuple[int, int, Dict[str, Any]]:
        """ Return lut-6 and levels after application of predefined sequence """
        sequence = [s + ";" for s in self.sequence]

        return get_design_prop(
            design_file=self.design_file,
            evaluator=evaluator,
            seq=sequence,
            print_stat_stages=None,
            verbose=verbose,
            new_op=np.any(["&" in s for s in self.sequence]) and not np.any(["&put" in s for s in self.sequence])
        )

    @staticmethod
    def seq_length() -> int:
        raise NotImplementedError()


class Resyn(BuildInSeq):
    aux_sequence = [
        'balance',
        'rewrite',
        'rewrite -z',
        'balance',
        'rewrite -z',
        'balance'
    ]

    def __init__(self, design_file: str, lut_inputs: int):
        """
            balance; rewrite; rewrite -z; balance; rewrite -z; balance
        """

        super().__init__(design_file=design_file)
        self.sequence = ["strash;"] + self.aux_sequence + [f"if -K {lut_inputs}"]

    @staticmethod
    def seq_length() -> int:
        return len(Resyn.aux_sequence) + 1


resyn2_seq = [
    'balance',
    'rewrite',
    'refactor',
    'balance',
    'rewrite',
    'rewrite -z',
    'balance',
    'refactor -z',
    'rewrite -z',
    'balance'
]

dresyn2_seq = [
    'balance',
    'drw',
    'drf',
    'balance',
    'drw',
    'drw -z',
    'balance',
    'drf -z',
    'drw -z',
    'balance',
]


class Resyn2(BuildInSeq):
    aux_sequence = resyn2_seq

    def __init__(self, design_file: str, lut_inputs: int):
        """
            balance; rewrite; refactor; balance; rewrite; rewrite –z; balance; refactor –z; rewrite –z; balance;
        """

        super().__init__(design_file=design_file)
        self.sequence = self.aux_sequence + [f"if -K {lut_inputs}"]

    @staticmethod
    def seq_length() -> int:
        return len(Resyn2.aux_sequence) + 1


class StrResyn2(BuildInSeq):
    aux_sequence = resyn2_seq

    def __init__(self, design_file: str, lut_inputs: int):
        """
            strash; balance; rewrite; refactor; balance; rewrite; rewrite –z; balance; refactor –z; rewrite –z; balance;
        """

        super().__init__(design_file=design_file)
        self.sequence = ["strash;"] + self.aux_sequence + [f"if -K {lut_inputs}"]

    @staticmethod
    def seq_length() -> int:
        return len(Resyn2.aux_sequence) + 1


class DResyn2Speedup(BuildInSeq):
    aux_sequence = dresyn2_seq

    def __init__(self, design_file: str, lut_inputs: int):
        """
            balance; drw; drf; balance; drw; drw –z; balance; drf –z; drw –z; balance;
            if -K ...; speedup; if -K ... -C 16 -F; speedup; if -K ... -C 16 -F; speedup; if -K ... -C 16 -F;
        """

        super().__init__(design_file=design_file)
        assert lut_inputs == 4, f"Usually this is used with -K 4 and not -K {lut_inputs}"
        self.sequence = ["strash;"] + self.aux_sequence + [f"if -K {lut_inputs}"]
        for _ in range(3):
            self.sequence += ["speedup", f"if -K {lut_inputs} -C 16 -F 2"]

    @staticmethod
    def seq_length() -> int:
        return len(DResyn2Speedup.aux_sequence) + 4


class InitDesign(BuildInSeq):
    aux_sequence = []

    def __init__(self, design_file: str, lut_inputs: int):
        """
            No action, evaluate initial design
        """

        super().__init__(design_file)
        self.sequence = self.aux_sequence + [f"if -K {lut_inputs}"]

    @staticmethod
    def seq_length() -> int:
        return len(InitDesign.aux_sequence) + 1


class StrInitDesign(BuildInSeq):
    aux_sequence = []

    def __init__(self, design_file: str, lut_inputs: int):
        """
            No action, evaluate initial design
        """

        super().__init__(design_file)
        self.sequence = ["strash;"] + self.aux_sequence + [f"if -K {lut_inputs}"]

    @staticmethod
    def seq_length() -> int:
        return len(InitDesign.aux_sequence) + 1


BUILD_IN_SEQ: Dict[str, Union[Type[InitDesign], Type[Resyn], Type[Resyn2]]] = dict(
    init=InitDesign,
    resyn=Resyn,
    resyn2=Resyn2,
    strash_resyn2=StrResyn2,
    str_init=StrInitDesign,
    dresyn2_speedup=DResyn2Speedup
)


def get_build_in_seq(build_in_seq_id: str):
    return BUILD_IN_SEQ[build_in_seq_id]


class RefObj:

    def __init__(self, design_file: str,
                 lut_inputs: int, evaluator: str,
                 ref_abc_seq: str,
                 n_eval_ref: int = 1):
        """
        Args:
            design_file: path to the design
            evaluator: *
            lut_inputs: number of LUT inputs (2 < num < 33)
            ref_abc_seq: sequence of operations to apply to initial design to get reference performance
            n_eval_ref: when exec time is important, ref is evaluated several times and the average time is reported
        """
        self.design_file = design_file
        self.lut_inputs = lut_inputs
        self.ref_abc_seq = ref_abc_seq
        self.evaluator = evaluator
        self.n_eval_ref = n_eval_ref
        self.design_name = os.path.basename(design_file).split('.')[0]
        assert self.n_eval_ref > 0

    def get_config(self) -> Dict[str, Any]:
        return dict(
            design_file=self.design_file,
            design_name=self.design_name,
            lut_inputs=self.lut_inputs,
            ref_abc_seq=self.ref_abc_seq,
            evaluator=self.evaluator,
            n_eval_ref=self.n_eval_ref
        )

    def ref_path(self) -> str:
        path_id = f"lut-{self.lut_inputs}"
        path_id += f'_{self.evaluator}'
        if self.n_eval_ref != 1:
            path_id += f"_n-eval-{self.n_eval_ref}"
        try:
            return os.path.join(get_results_storage_path_root(), 'refs', self.ref_abc_seq, path_id, self.design_name)
        except Exception:
            print(get_results_storage_path_root(), 'refs', self.ref_abc_seq, path_id, self.design_name)
            raise

    def get_refs(self, ignore_existing: bool = False) -> Tuple[float, float, float]:
        if os.path.exists(os.path.join(self.ref_path(), 'refs.pkl')) and not ignore_existing:
            refs = load_w_pickle(self.ref_path(), 'refs.pkl')
        else:
            ref_1, ref_2 = None, None
            exec_times = []
            for _ in range(self.n_eval_ref):
                biseq_cl = BUILD_IN_SEQ[self.ref_abc_seq]
                biseq = biseq_cl(design_file=self.design_file, lut_inputs=self.lut_inputs)

                ref_1, ref_2, extra_info = biseq.fpga(verbose=False, evaluator=self.evaluator)
                exec_times.append(extra_info['exec_time'])
            os.makedirs(self.ref_path(), exist_ok=True)
            refs = dict(ref_1=ref_1, ref_2=ref_2, config=self.get_config(), exec_time=np.mean(exec_times))
            if not ignore_existing:
                save_w_pickle(refs, self.ref_path(), 'refs.pkl')
        return refs['ref_1'], refs['ref_2'], refs["exec_time"]


def get_ref(design_file: str, lut_inputs: int, ref_abc_seq: str, evaluator: str, n_eval_ref: int) \
        -> Tuple[float, float, float]:
    """ Return either area and delay or lut and levels obtained when applying a given sequence on a given
        design file.
    """

    ref_obj = RefObj(design_file=design_file, lut_inputs=lut_inputs,
                     ref_abc_seq=ref_abc_seq,
                     evaluator=evaluator,
                     n_eval_ref=n_eval_ref)

    ref_1, ref_2, exec_time = ref_obj.get_refs()

    return ref_1, ref_2, exec_time


if __name__ == '__main__':
    print('; '.join(resyn2_seq))

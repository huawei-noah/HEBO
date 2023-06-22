import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Type, Union, Set, Optional

################################
# Logic optimization operators
################################
import numpy as np
import torch

from mcbo.tasks.eda_seq_opt.utils.utils import get_circuits_path_root

LUT_MAPPING_HYPERPARAMS: Set[str] = set()


class OperatorType(ABC):
    num_id: int
    typename: str

    def __repr__(self):
        return self.typename

    def __eq__(self, other):
        return other.typename == self.typename


class PreMappingOperatorType(OperatorType):
    num_id = 0
    typename = "pre-mapping"


class MappingOperatorType(OperatorType):
    num_id = 1
    typename = "mapping"


class PostMappingOperatorType(OperatorType):
    num_id = 2
    typename = "post-mapping"


class NoneOperatorType(OperatorType):
    num_id = 3
    typename = "none"


class StrashOperatorType(OperatorType):
    num_id = 4
    typename = "strash"


PRE_MAPPING_OPERATOR_TYPE = PreMappingOperatorType()
MAPPING_OPERATOR_TYPE = MappingOperatorType()
POST_MAPPING_OPERATOR_TYPE = PostMappingOperatorType()
STRASH_OPERATOR_TYPE = StrashOperatorType()
NONE_OPERATOR_TYPE = NoneOperatorType()


class OperatorHyperparam:

    def __init__(self, op_hyper_id: str, str_expr: str):
        self.op_hyper_id = op_hyper_id
        self.str_expr = str_expr


class OperatorHyperparamSwitch(OperatorHyperparam):

    def __init__(self, op_hyper_id: str, flag: str, activate: bool):
        super().__init__(op_hyper_id=op_hyper_id, str_expr=f"{flag}" if activate else "")


class OperatorHyperparamVal(OperatorHyperparam):

    def __init__(self, op_hyper_id: str, flag: str, value: Any):
        super().__init__(op_hyper_id=op_hyper_id, str_expr=f"{flag} {value}")


class Operator(ABC):
    """ abc operator """
    hyperparams: Dict[str, Union[Tuple[Type[OperatorHyperparam], str], Tuple[Type[OperatorHyperparam], str, Any]]]
    op_id: str
    requires_rec_start3: bool

    def __init__(self, op_type: OperatorType, **kwargs):
        """
        Args:
            op_id: operator id
            op_type: type of the operator
            **kwargs: dictionary with hyperparams values of the operators
        """
        self.op_type = op_type
        self.kw_str = ""
        for k in self.hyperparams:
            self.kw_str += self.get_hyperparam_str(k, **kwargs)

    def get_hyperparam_str(self, k: str, **kwargs):
        """ Get hyperparam (flag + val) given the key `k` """
        v = self.hyperparams[k]
        if issubclass(v[0], OperatorHyperparamSwitch):
            if k in kwargs:
                activate = kwargs.get(k)
            else:
                activate = v[2]
            return v[0](op_hyper_id=k, flag=v[1], activate=activate).str_expr + " "
        elif issubclass(v[0], OperatorHyperparamVal):
            if k in kwargs:
                value = kwargs.get(k)
            else:
                value = v[2]
            return v[0](op_hyper_id=k, flag=v[1], value=value).str_expr + " "
        else:
            raise TypeError()

    @abstractmethod
    def op_str(self) -> str:
        """ String used to apply the operator """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def check() -> None:
        """ Check that operator can be used """
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.op_id} -> {self.op_str()} ({self.op_type})"

    def __eq__(self, other):
        return self.op_id == other.op_id

    def __hash__(self):
        return hash(self.op_id)


class OperatorSimple(Operator, ABC):
    """ Operator for which op_str = op_id + ;  """

    def op_str(self) -> str:
        """ String used to apply the operator """
        return f"{self.op_id} {self.kw_str} ;"


class Operator9(Operator, ABC):

    def __init__(self, op_type: OperatorType, use_get: bool, use_put: bool, **kwargs):
        """
        Args:
            op_type: type of the operator
            **kwargs: dictionary with hyperparams values of the operators
        """
        super().__init__(op_type=op_type, **kwargs)
        self.use_get = use_get
        self.use_put = use_put

    def op_str(self) -> str:
        """ String used to apply the operator """
        op_str = ""
        if self.use_get:
            op_str += "&get -n; "
        op_str += f'{self.op_id} {self.kw_str}; '
        if self.use_put:
            op_str += '&put;'
        return op_str


class PrintStats(OperatorSimple):
    @staticmethod
    def check() -> None:
        pass

    hyperparams = {}
    op_id = "print_stats"

    def __init__(self, **kwargs):
        super().__init__(op_type=NONE_OPERATOR_TYPE, **kwargs)


# -------------------------------- Pre-mapping operators -------------------------------- #

REC_START_3_PATH = os.path.join(get_circuits_path_root(), 'rec6Lib_final_filtered3_recanon.aig')


class RecStart3(Operator):
    hyperparams = {
        "rec_start3 K": (OperatorHyperparamVal, '-K', 6),
        "rec_start3 C": (OperatorHyperparamVal, '-C', 4096),
    }
    op_id = 'rec_start3_lms'

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, **kwargs)

    def op_str(self) -> str:
        return f"rec_start3 {self.kw_str} {REC_START_3_PATH};"

    @staticmethod
    def check() -> None:
        assert os.path.exists(REC_START_3_PATH)


class Rewrite(OperatorSimple):
    hyperparams = {
        'rewrite -l': (OperatorHyperparamSwitch, '-l', 0),
        'rewrite -z': (OperatorHyperparamSwitch, '-z', 0)
    }
    op_id = 'rewrite'
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, **kwargs)

    @staticmethod
    def check() -> None:
        pass


class RewriteWoZ(Operator):
    hyperparams = {
        'rewrite_wo_z -l': (OperatorHyperparamSwitch, '-l', 0),
    }
    op_id = 'rewrite_wo_z'
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, **kwargs)

    @staticmethod
    def check() -> None:
        pass

    def op_str(self) -> str:
        return f"rewrite {self.kw_str};"


class RewriteZ(Operator):
    hyperparams = {
        'rewrite_w_z -l': (OperatorHyperparamSwitch, '-l', 0),
    }

    op_id = 'rewrite_w_z'
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, **kwargs)

    @staticmethod
    def check() -> None:
        pass

    def op_str(self) -> str:
        return f"rewrite -z {self.kw_str};"


class DRewrite(OperatorSimple):
    hyperparams = {
        'drw -l': (OperatorHyperparamSwitch, '-l', 0),
        'drw -z': (OperatorHyperparamSwitch, '-z', 0),
        'drw C': (OperatorHyperparamVal, '-C', 8),
        'drw N': (OperatorHyperparamVal, '-N', 5)
    }
    op_id = 'drw'
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, **kwargs)

    @staticmethod
    def check() -> None:
        pass


class DRewriteWoZ(Operator):
    hyperparams = {
        'drw_wo_z -l': (OperatorHyperparamSwitch, '-l', 0),
        'drw_wo_z C': (OperatorHyperparamVal, '-C', 8),
        'drw_wo_z N': (OperatorHyperparamVal, '-N', 5)
    }
    op_id = 'drw_wo_z'
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, **kwargs)

    @staticmethod
    def check() -> None:
        pass

    def op_str(self) -> str:
        return f"drw {self.kw_str};"


class DRewriteZ(Operator):
    hyperparams = {
        'drw_w_z -l': (OperatorHyperparamSwitch, '-l', 0),
        'drw_w_z C': (OperatorHyperparamVal, '-C', 8),
        'drw_w_z N': (OperatorHyperparamVal, '-N', 5)
    }
    op_id = 'drw_w_z'
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, **kwargs)

    @staticmethod
    def check() -> None:
        pass

    def op_str(self) -> str:
        return f"drw -z {self.kw_str};"


class Refactor(OperatorSimple):
    hyperparams = {
        'refactor N': (OperatorHyperparamVal, '-N', 10),
        'refactor -z': (OperatorHyperparamSwitch, '-z', 0),
        'refactor -l': (OperatorHyperparamSwitch, '-l', 0)
    }

    op_id = 'refactor'
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, **kwargs)

    @staticmethod
    def check() -> None:
        pass


class RefactorWoZ(Operator):
    hyperparams = {
        'refactor_wo_z N': (OperatorHyperparamVal, '-N', 10),
        'refactor_wo_z -l': (OperatorHyperparamSwitch, '-l', 0)
    }

    op_id = 'refactor_wo_z'
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, **kwargs)

    @staticmethod
    def check() -> None:
        pass

    def op_str(self) -> str:
        return f"refactor {self.kw_str};"


class RefactorZ(Operator):
    hyperparams = {
        'refactor_w_z N': (OperatorHyperparamVal, '-N', 10),
        'refactor_w_z -l': (OperatorHyperparamSwitch, '-l', 0)
    }

    op_id = 'refactor_w_z'
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, **kwargs)

    @staticmethod
    def check() -> None:
        pass

    def op_str(self) -> str:
        return f"refactor -z {self.kw_str};"


class DRefactor(OperatorSimple):
    hyperparams = {
        'drf K': (OperatorHyperparamVal, '-K', 12),
        'drf C': (OperatorHyperparamVal, '-C', 5),
        'drf -z': (OperatorHyperparamSwitch, '-z', 0),
        'drf -l': (OperatorHyperparamSwitch, '-l', 0)
    }

    op_id = 'drf'
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, **kwargs)

    @staticmethod
    def check() -> None:
        pass


class DRefactorZ(OperatorSimple):
    hyperparams = {
        'drf_w_z K': (OperatorHyperparamVal, '-K', 12),
        'drf_w_z C': (OperatorHyperparamVal, '-C', 5),
        'drf_w_z -l': (OperatorHyperparamSwitch, '-l', 0)
    }

    op_id = 'drf_w_z'
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, **kwargs)

    @staticmethod
    def check() -> None:
        pass

    def op_str(self) -> str:
        return f"drf -z {self.kw_str};"


class DRefactorWoZ(OperatorSimple):
    hyperparams = {
        'drf_wo_z K': (OperatorHyperparamVal, '-K', 12),
        'drf_wo_z C': (OperatorHyperparamVal, '-C', 5),
        'drf_wo_z -l': (OperatorHyperparamSwitch, '-l', 0)
    }

    op_id = 'drf_wo_z'
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, **kwargs)

    @staticmethod
    def check() -> None:
        pass

    def op_str(self) -> str:
        return f"drf {self.kw_str};"


class Resub(OperatorSimple):
    hyperparams = {
        'resub K': (OperatorHyperparamVal, '-K', 8),
        'resub F': (OperatorHyperparamVal, '-F', 0),
        'resub N': (OperatorHyperparamVal, '-N', 1),
        'resub -z': (OperatorHyperparamSwitch, '-z', 0),
        'resub -l': (OperatorHyperparamSwitch, '-l', 0)
    }

    op_id = 'resub'
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, **kwargs)

    @staticmethod
    def check() -> None:
        pass


class ResubZ(Operator):
    hyperparams = {
        'resub_w_z K': (OperatorHyperparamVal, '-K', 8),
        'resub_w_z F': (OperatorHyperparamVal, '-F', 0),
        'resub_w_z N': (OperatorHyperparamVal, '-N', 1),
        'resub_w_z -l': (OperatorHyperparamSwitch, '-l', 0)
    }

    op_id = 'resub_w_z'
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, **kwargs)

    @staticmethod
    def check() -> None:
        pass

    def op_str(self) -> str:
        return f"resub -z {self.kw_str};"


class ResubWoZ(Operator):
    hyperparams = {
        'resub_wo_z K': (OperatorHyperparamVal, '-K', 8),
        'resub_wo_z F': (OperatorHyperparamVal, '-F', 0),
        'resub_wo_z N': (OperatorHyperparamVal, '-N', 1),
        'resub_wo_z -l': (OperatorHyperparamSwitch, '-l', 0)
    }

    op_id = 'resub_wo_z'
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, **kwargs)

    @staticmethod
    def check() -> None:
        pass

    def op_str(self) -> str:
        return f"resub {self.kw_str};"


class Balance(OperatorSimple):
    hyperparams = {
        'balance -l': (OperatorHyperparamSwitch, '-l', 0),
        'balance -d': (OperatorHyperparamSwitch, '-d', 0),
        'balance -s': (OperatorHyperparamSwitch, '-s', 0),
        'balance -x': (OperatorHyperparamSwitch, '-x', 0),
    }

    op_id = 'balance'
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, **kwargs)

    @staticmethod
    def check() -> None:
        pass


class Fraig(Operator9):
    hyperparams = {
        'fraig -r': (OperatorHyperparamSwitch, '-r', 0),
    }
    op_id = 'fraig'
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, use_get=True, use_put=True, **kwargs)

    @staticmethod
    def check() -> None:
        pass


class Sopb(Operator9):
    hyperparams = {
        '&sopb C': (OperatorHyperparamVal, '-C', 8),
    }
    op_id = '&sopb'
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, use_get=True, use_put=True, **kwargs)

    @staticmethod
    def check() -> None:
        pass


class SimpleSopb(Operator):
    hyperparams = {
        'sopb C': (OperatorHyperparamVal, '-C', 8),
        'sopb K': (OperatorHyperparamVal, '-K', 6),
    }
    op_id = "sopb"
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, **kwargs)

    def op_str(self) -> str:
        return f"if -g {self.kw_str};"

    @staticmethod
    def check() -> None:
        pass


LUT_MAPPING_HYPERPARAMS.add("sopb K")


class Dsdb(Operator9):
    hyperparams = {
        '&dsdb C': (OperatorHyperparamVal, '-C', 8),
    }
    op_id = '&dsdb'
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, use_get=True, use_put=True, **kwargs)

    @staticmethod
    def check() -> None:
        pass


class SimpleDsdb(Operator):
    hyperparams = {
        'dsdb C': (OperatorHyperparamVal, '-C', 8),
        'dsdb K': (OperatorHyperparamVal, '-K', 6),
    }
    op_id = "dsdb"
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, **kwargs)

    def op_str(self) -> str:
        return f"if -x {self.kw_str};"

    @staticmethod
    def check() -> None:
        pass


LUT_MAPPING_HYPERPARAMS.add("dsdb K")


class Blut(Operator9):
    hyperparams = {
        '&blut C': (OperatorHyperparamVal, '-C', 8),
        '&blut -a': (OperatorHyperparamSwitch, '-a', 0),
    }
    requires_rec_start3 = False

    op_id = '&blut'

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, use_get=True, use_put=True, **kwargs)

    @staticmethod
    def check() -> None:
        pass


class LMS(Operator):
    hyperparams = {
        "LMS C": (OperatorHyperparamVal, '-C', 8),
        "LMS K": (OperatorHyperparamVal, '-K', 6)
    }
    op_id = "LMS"
    requires_rec_start3 = True

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, **kwargs)

    def op_str(self) -> str:
        return f"if -y {self.kw_str};"

    @staticmethod
    def check() -> None:
        if not os.path.exists(REC_START_3_PATH):
            raise FileNotFoundError(f"To use LMS you need file: {REC_START_3_PATH}")


LUT_MAPPING_HYPERPARAMS.add("LMS K")


class Strash(OperatorSimple):
    hyperparams = {}
    op_id = "strash"
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, **kwargs)

    @staticmethod
    def check() -> None:
        pass


PRE_MAPPING_OPERATORS: Dict[str, Type[Operator]] = {
    OP.op_id: OP for OP in
    [
        Rewrite, RewriteZ, RewriteWoZ, DRewrite, DRewriteZ, DRewriteWoZ,
        Refactor, RefactorZ, RefactorWoZ, DRefactor, DRefactorZ, DRefactorWoZ,
        Resub, ResubZ, ResubWoZ,
        Balance, Fraig, Sopb, SimpleSopb, Dsdb, SimpleDsdb, Blut, LMS, RecStart3, Strash,
    ]
}

BOILS_PRE_MAPPING_OPERATORS: List[Type[Operator]] = [
    RewriteWoZ, RewriteZ, RefactorWoZ, RefactorZ, Resub, Balance, Blut, Sopb, Dsdb, Fraig
]


# -------------------------------- Mapping operators -------------------------------- #

class If(OperatorSimple):
    hyperparams = {
        'if K': (OperatorHyperparamVal, '-K', 6),
        'if C': (OperatorHyperparamVal, '-C', 8),
        'if F': (OperatorHyperparamVal, '-F', 1),
        'if A': (OperatorHyperparamVal, '-A', 2),
    }

    op_id = "if"
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(MAPPING_OPERATOR_TYPE, **kwargs)

    @staticmethod
    def check() -> None:
        pass


LUT_MAPPING_HYPERPARAMS.add("if K")


class IfA(OperatorSimple):
    hyperparams = {
        'if -a K': (OperatorHyperparamVal, '-K', 6),
        'if -a C': (OperatorHyperparamVal, '-C', 8),
        'if -a F': (OperatorHyperparamVal, '-F', 1),
        'if -a A': (OperatorHyperparamVal, '-A', 2),
    }

    op_id = "if -a"
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(MAPPING_OPERATOR_TYPE, **kwargs)

    @staticmethod
    def check() -> None:
        pass


LUT_MAPPING_HYPERPARAMS.add("if -a K")

MAPPING_OPERATORS: Dict[str, Type[Operator]] = {
    OP.op_id: OP for OP in [If, IfA]
}

BOILS_MAPPING_OPERATORS: List[Type[Operator]] = [
    If, IfA
]


# -------------------------------- Post-mapping operators -------------------------------- #


class SpeedupIf(Operator):
    hyperparams = {
        'speedup_if speedup P': (OperatorHyperparamVal, '-P', 5),
        'speedup_if speedup N': (OperatorHyperparamVal, '-N', 2),
        'speedup_if if K': (OperatorHyperparamVal, '-K', 6),
        'speedup_if if F': (OperatorHyperparamVal, '-F', 1),
        'speedup_if if C': (OperatorHyperparamVal, '-C', 8),
    }
    op_id = "speedup_if"
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(POST_MAPPING_OPERATOR_TYPE, **kwargs)
        self.speedup_kw_str = ""
        self.if_kw_str = ""
        for k in self.hyperparams:
            if k.split()[1] == "if":
                self.if_kw_str += self.get_hyperparam_str(k, **kwargs)
            elif k.split()[1] == "speedup":
                self.speedup_kw_str += self.get_hyperparam_str(k, **kwargs)
            else:
                raise ValueError

    def op_str(self) -> str:
        """ String used to apply the operator """
        return f"speedup {self.speedup_kw_str}; if {self.if_kw_str};"

    @staticmethod
    def check() -> None:
        pass


LUT_MAPPING_HYPERPARAMS.add("speedup_if if K")


class Mfs2(OperatorSimple):
    hyperparams = {
        'mfs2 W': (OperatorHyperparamVal, '-W', 2),
        'mfs2 M': (OperatorHyperparamVal, '-M', 300),
        'mfs2 D': (OperatorHyperparamVal, '-D', 20),
        'mfs2 -a': (OperatorHyperparamSwitch, '-a', 0),
        'mfs2 -e': (OperatorHyperparamSwitch, '-e', 0),
    }
    op_id = "mfs2"
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(POST_MAPPING_OPERATOR_TYPE, **kwargs)

    @staticmethod
    def check() -> None:
        pass


class Lutpack(OperatorSimple):
    hyperparams = {
        "lutpack N": (OperatorHyperparamVal, '-N', 4),
        "lutpack S": (OperatorHyperparamVal, '-S', 0),
        "lutpack -z": (OperatorHyperparamSwitch, '-z', 0),
    }

    op_id = "lutpack"
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(POST_MAPPING_OPERATOR_TYPE, **kwargs)

    @staticmethod
    def check() -> None:
        pass


class Mfs2Lutpack(Operator):
    hyperparams = {
        'mfs2_lutpack mfs2 W': (OperatorHyperparamVal, '-W', 2),
        'mfs2_lutpack mfs2 M': (OperatorHyperparamVal, '-M', 300),
        'mfs2_lutpack mfs2 D': (OperatorHyperparamVal, '-D', 20),
        'mfs2_lutpack mfs2 -a': (OperatorHyperparamSwitch, '-a', 0),
        'mfs2_lutpack mfs2 -e': (OperatorHyperparamSwitch, '-e', 0),
        "mfs2_lutpack lutpack N": (OperatorHyperparamVal, '-N', 4),
        "mfs2_lutpack lutpack S": (OperatorHyperparamVal, '-S', 0),
        "mfs2_lutpack lutpack -z": (OperatorHyperparamSwitch, '-z', 0),
    }
    op_id = "mfs2_lutpack"
    requires_rec_start3 = False

    def __init__(self, **kwargs):
        super().__init__(POST_MAPPING_OPERATOR_TYPE, **kwargs)
        self.mfs2_kw_str = ""
        self.lutpack_kw_str = ""
        for k in self.hyperparams:
            if k.split()[1] == "mfs2":
                self.mfs2_kw_str += self.get_hyperparam_str(k, **kwargs)
            elif k.split()[1] == 'lutpack':
                self.lutpack_kw_str += self.get_hyperparam_str(k, **kwargs)
            else:
                raise ValueError

    def op_str(self) -> str:
        """ String used to apply the operator """
        return f"mfs2 {self.mfs2_kw_str}; lutpack {self.lutpack_kw_str};"

    @staticmethod
    def check() -> None:
        pass


POST_MAPPING_OPERATORS: Dict[str, Type[Operator]] = {
    OP.op_id: OP for OP in [SpeedupIf, Mfs2, Lutpack, Mfs2Lutpack]
}

BOILS_POST_MAPPING_OPERATORS: List[Type[Operator]] = [
    SpeedupIf, Mfs2Lutpack
]


# -------------------------------- New operators Pre-mapping ------------------------ #

class PrintStats9(OperatorSimple):
    @staticmethod
    def check() -> None:
        pass

    hyperparams = {}
    op_id = "&ps"

    def __init__(self, **kwargs):
        super().__init__(op_type=NONE_OPERATOR_TYPE, **kwargs)


class Strash9(Operator):
    hyperparams = {}

    requires_rec_start3 = False
    op_id = '&st'

    def __init__(self, **kwargs):
        super().__init__(PRE_MAPPING_OPERATOR_TYPE, **kwargs)

    def op_str(self) -> str:
        return "&st;"

    @staticmethod
    def check() -> None:
        pass


# -------------------------------- ----------------- -------------------------------- #

OPERATORS = PRE_MAPPING_OPERATORS.copy()
OPERATORS.update(MAPPING_OPERATORS)
OPERATORS.update(POST_MAPPING_OPERATORS)


def get_operator(op_id: str):
    return OPERATORS[op_id]


class SeqOperatorsPattern:
    """ Define the sequence of operators
     (0: pre-mapping, 1: mapping, 2: post-mapping, -1: any, -2: mapping or post-mapping, -3: pre-mapping or mapping)
    """
    VALID_TYPE_VALUES = {-2, -1, 0, 1, 2}
    INVALID_SEQUENCE = {
        (1, 1),  # 2 mapping operations in a row
        (0, 2),  # post_mapping following pre-mapping
        (2, 1),  # mapping following post-mapping
    }

    def __init__(self, pattern: List[int]):
        self.pattern = pattern
        assert set(self.pattern).issubset(self.VALID_TYPE_VALUES), set(self.pattern)
        self.contains_free = np.any(np.array(self.pattern) < 0)
        if self.contains_free:
            self.n_print_stats = 1 + len(self.pattern)
        else:
            self.n_print_stats = self.get_n_print_stats(pattern=self.pattern)

    def __len__(self):
        return len(self.pattern)

    def __eq__(self, other):
        if not isinstance(other, SeqOperatorsPattern):
            return False
        return np.all(np.array(self.pattern) == np.array(other.pattern))

    def is_valid(self, seq_op_type):
        for op_type, expected_type in zip([seq_op_type, self.pattern]):
            if not self.op_type_is_valid_type(op_type=op_type, expected_type=expected_type):
                return False
        return True

    @staticmethod
    def get_valid_types(op_type: int) -> List[int]:
        if op_type == -3:
            return [0, 1]
        if op_type == -2:
            return [1, 2]
        if op_type == -1:
            return [0, 1, 2]
        return [op_type]

    @staticmethod
    def op_type_is_valid_type(op_type: int, expected_type: int):
        assert op_type in SeqOperatorsPattern.VALID_TYPE_VALUES
        return op_type in SeqOperatorsPattern.get_valid_types(op_type=expected_type)

    @staticmethod
    def get_n_print_stats(pattern: List[int]):
        """
        `print_stats` should be called initially, and after each mapping and post-mapping operations
        If the sequence contains free operator (negative values), it's not possible to know the number of print_stats
        in advance
        """
        assert np.all(np.array(pattern) >= 0), f"{pattern} contains freedom"
        return 1 + pattern.count(MAPPING_OPERATOR_TYPE.num_id) + pattern.count(POST_MAPPING_OPERATOR_TYPE.num_id)

    @staticmethod
    def find_type_seq(possible_types: List[List[int]], previous_type: int) -> Optional[List[int]]:
        if len(possible_types) == 1:
            for possible_type in possible_types[0]:
                if (previous_type, possible_type) not in SeqOperatorsPattern.INVALID_SEQUENCE:
                    return [possible_type]
            return None

        for possible_type in possible_types[0]:
            if (previous_type, possible_type) not in SeqOperatorsPattern.INVALID_SEQUENCE:
                seq = SeqOperatorsPattern.find_type_seq(possible_types=possible_types[1:], previous_type=possible_type)
                if seq is not None:
                    return [possible_type] + seq

        return None

    def sample(self, op_ind_per_type_dic: Dict[int, List[int]]) -> Union[np.ndarray, List[int]]:
        possible_types = [list(np.random.permutation(self.get_valid_types(op_type))) for op_type in self.pattern]
        type_seq = self.find_type_seq(possible_types=possible_types,
                                      previous_type=-1)  # output sequence will follow this sequence type
        if type_seq is None:
            raise ValueError("Expected pattern cannot be obtained")
        seq = []
        for i in range(len(type_seq)):
            seq.append(np.random.choice(op_ind_per_type_dic[type_seq[i]]))

        return seq

    def get_valid_ops_at_dim(self, current_seq: torch.Tensor, dim: int, op_ind_per_type_dic: Dict[int, List[int]],
                             op_to_type_dic: Dict[int, int]) -> List[int]:
        valid_types = self.get_valid_types(self.pattern[dim])
        to_remove_types = set()
        if dim > 0:
            previous_type = op_to_type_dic[current_seq[dim - 1]]  # type of previous op
            for v_type in valid_types:
                if (previous_type, v_type) in self.INVALID_SEQUENCE:
                    to_remove_types.add(v_type)
        if dim < (len(self) - 1):
            next_type = op_to_type_dic[current_seq[dim + 1]]  # type of next op
            for v_type in valid_types:
                if (v_type, next_type) in self.INVALID_SEQUENCE:
                    to_remove_types.add(v_type)

        for t in to_remove_types:
            valid_types.remove(t)

        valid_op_ind = set()
        for t in valid_types:
            valid_op_ind.update(op_ind_per_type_dic[t])

        return list(valid_op_ind)

    def not_mapped_at_the_end(self) -> bool:
        """ Return True if the pattern is such that it is sure that the circuit won't be a mapped netlist at the end """
        return self.pattern[-1] == PRE_MAPPING_OPERATOR_TYPE.num_id


SEQ_OPERATOR_PATTERNS: Dict[str, SeqOperatorsPattern] = {
    'basic': SeqOperatorsPattern([0] * 20),
    'basic_w_post_map': SeqOperatorsPattern(([0] * 7 + [1] + [2, 2]) * 2),
}


def get_seq_operators_pattern(seq_operators_pattern: Optional[str]) -> Optional[SeqOperatorsPattern]:
    if seq_operators_pattern:
        return SEQ_OPERATOR_PATTERNS[seq_operators_pattern]
    return None


class OperatorSpace:

    def __init__(self, pre_mapping_operators: List[Type[Operator]], mapping_operators: List[Type[Operator]],
                 post_mapping_operators: List[Type[Operator]]):
        self.post_mapping_operators = post_mapping_operators
        self.mapping_operators = mapping_operators
        self.pre_mapping_operators = pre_mapping_operators
        self.all_operators: List[Type[Operator]] = self._get_all_operators_list()
        assert len(self.all_operators) == len(self.all_operators), "Duplicated operators"

    def _get_all_operators_list(self) -> List[Type[Operator]]:
        return self.pre_mapping_operators + self.mapping_operators + self.post_mapping_operators

    def __len__(self):
        return len(self.all_operators)

    def check(self):
        """ Check if all operators can be applied
        - For LMS: need to check that rec6Lib_final_filtered3_recanon.aig is available
        """
        for op in self.all_operators:
            op.check()


OPERATOR_SPACES: Dict[str, OperatorSpace] = {
    'basic': OperatorSpace(
        pre_mapping_operators=BOILS_PRE_MAPPING_OPERATORS,
        mapping_operators=BOILS_MAPPING_OPERATORS,
        post_mapping_operators=BOILS_POST_MAPPING_OPERATORS
    ),
}


def is_lut_mapping_hyperparam(param_name: str):
    return param_name in LUT_MAPPING_HYPERPARAMS


def get_operator_space(operator_space_id: str) -> OperatorSpace:
    return OPERATOR_SPACES[operator_space_id]


def make_operator_sequence_valid(operator_sequence: List[Operator], final_mapping_op: Optional[Operator] = None) -> Tuple[List[Operator], bool]:
    """
    Add `strash` and `rec_start3` operations if needs be to make the operator sequence valid
    """
    seq_of_new_ops = False
    for op in operator_sequence:
        if "9" in op.__class__.__name__:
            seq_of_new_ops = True
    if seq_of_new_ops and not np.all(["9" in op.__class__.__name__ for op in operator_sequence]):
        raise ValueError("Should either have only new ops or only old ops")
    if seq_of_new_ops:
        return make_new_operator_sequence_valid(operator_sequence=operator_sequence), seq_of_new_ops
    else:
        return make_old_operator_sequence_valid(operator_sequence=operator_sequence,
                                                final_mapping_op=final_mapping_op), seq_of_new_ops


def make_old_operator_sequence_valid(operator_sequence: List[Operator], final_mapping_op: Optional[Operator] = None) -> List[
    Operator]:
    """
    Add `strash` and `rec_start3` operations if needs be to make the operator sequence valid
    """
    final_action_sequence: List[Operator] = [RecStart3()]
    previous_action_type = MAPPING_OPERATOR_TYPE
    include_rec_start3: bool = False
    for i, operator in enumerate(operator_sequence):
        if previous_action_type in [POST_MAPPING_OPERATOR_TYPE, MAPPING_OPERATOR_TYPE]:
            final_action_sequence.append(PrintStats())
        if operator.requires_rec_start3:
            include_rec_start3 = True
        if operator.op_type == PRE_MAPPING_OPERATOR_TYPE and previous_action_type in [MAPPING_OPERATOR_TYPE,
                                                                                      POST_MAPPING_OPERATOR_TYPE]:
            final_action_sequence.append(Strash())
        previous_action_type = operator.op_type
        final_action_sequence.append(operator)
    if not include_rec_start3:
        # remove first REC_START_3 action
        final_action_sequence = final_action_sequence[1:]
    if final_mapping_op is not None and previous_action_type not in [MAPPING_OPERATOR_TYPE, POST_MAPPING_OPERATOR_TYPE]:
        final_action_sequence.append(final_mapping_op)
    final_action_sequence.append(PrintStats())
    return final_action_sequence


def make_new_operator_sequence_valid(operator_sequence: List[Operator]) -> List[Operator]:
    """
    Add `&st` and `rec_start3` operations if needs be to make the operator sequence valid
    """
    final_action_sequence: List[Operator] = [RecStart3()]
    previous_action_type = MAPPING_OPERATOR_TYPE
    include_rec_start3: bool = False
    for i, operator in enumerate(operator_sequence):
        if previous_action_type in [POST_MAPPING_OPERATOR_TYPE, MAPPING_OPERATOR_TYPE]:
            final_action_sequence.append(PrintStats9())
        if operator.requires_rec_start3:
            include_rec_start3 = True
        if operator.op_type == PRE_MAPPING_OPERATOR_TYPE and previous_action_type in [MAPPING_OPERATOR_TYPE,
                                                                                      POST_MAPPING_OPERATOR_TYPE]:
            final_action_sequence.append(Strash9())
        previous_action_type = operator.op_type
        final_action_sequence.append(operator)
    if not include_rec_start3:
        # remove first REC_START_3 action
        final_action_sequence = final_action_sequence[1:]
    assert not include_rec_start3, "Should not use LMS"
    final_action_sequence.append(PrintStats9())
    return final_action_sequence

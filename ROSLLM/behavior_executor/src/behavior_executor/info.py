from enum import Enum


class Info(Enum):
    # OK is 0
    # Positive means warning
    # Negative means failure

    # no problems, behavior executed as expected
    OK = 0

    # nothing happened
    NULL_BEHAVIOR = 1

    # compilation of the behavior failed
    FAILED_TO_COMPILE = -1

    # an atomic action was referenced that does not exist
    ATOMIC_ACTION_UNAVAILABLE = -2

    # code import error
    CODE_IMPORT_ERROR = -3

    # execution error when running code
    CODE_EXECUTION_ERROR = -4

    # Saving code failed
    SAVE_CODE_FAILED = -5

    # exception thrown during execution of atomic action
    ERROR_ATOMIC_ACTION = -6

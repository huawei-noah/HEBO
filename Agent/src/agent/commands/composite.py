from functools import partial

from agent.commands import ConsiderAction
from agent.commands import ConsistencyOnDiverseActions
from agent.commands import ExecutePlannedAction
from agent.commands import ReflectOnPlannedAction
from agent.commands import SequentialFlow
from agent.memory import MemKey

Act = partial(
    SequentialFlow,
    name="Act",
    description="Finalize the action that will be performed in the environment",
    sequence=[ConsiderAction(), ExecutePlannedAction()],
)

ZeroStepReflect = partial(
    SequentialFlow,
    name="zero_step_reflect",
    description="Ask the model if its sure of its answer and confirm it.",
    sequence=[ConsiderAction(), ReflectOnPlannedAction(), ExecutePlannedAction()],
)

SelfConsistencyAct = partial(
    SequentialFlow,
    name="self_consistency_act",
    description="Run CoT multiple times and select the most consistent of answers.",
    sequence=5 * [ConsiderAction(output_keys={"output_mem_key": MemKey.NEXT_PLANNED_ACTION_DIVERSE}, append=True)]
    + [ConsistencyOnDiverseActions(), ExecutePlannedAction()],
)

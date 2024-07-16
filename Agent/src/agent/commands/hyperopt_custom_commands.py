import datetime
import os.path
from datetime import time
from typing import Union, Any, Dict, Tuple

import numpy as np
import pandas as pd
from hebo.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

from agent.agents import LLMAgent, safe_parsing_chat_completion
from agent.commands import SequentialFlow, LoopFlow, Act
from agent.commands.core import HumanTakeoverCommand
from agent.commands.flows import PARSE_FUNC_MAP
from agent.memory import MemKey
from agent.utils.hyperopt_utils import assign_hyperopt, wrap_code
from agent.utils.utils import get_path_to_python, extract_json


class HyperOpt(SequentialFlow):
    def __init__(
        self,
        summarize_code_prompt_template: str,
        loop_flow_prompt_template: str,
        error_instruct_prompt_template: str,
        search_space_prompt_template: str,
        workspace_path: str,
        bo_steps: int,
        max_repetitions: int = -1,
        max_retries: int = 5,
        use_error_instructions: bool = False,
    ):
        """
        Flow = SequentialFlow
                - Summarize code elements
                - LoopFlow                  (outer loop on search spaces)
                   SequentialFlow
                     - SuggestSearchSpace
                     - RunHyperopt
        """
        summarize_cmd = SummarizeCode(
            required_prompt_templates={"summarize_code_prompt_template": summarize_code_prompt_template},
            max_retries=max_retries,
        )
        error_instruct_cmd = ErrorInstruct(
            required_prompt_templates={"error_instruct_template": error_instruct_prompt_template},
            max_retries=max_retries,
        )
        suggest_cmd = SuggestSearchSpace(
            required_prompt_templates={"search_space_prompt_template": search_space_prompt_template},
            max_retries=max_retries,
        )
        hyperopt_cmd = RunHyperOpt(workspace_path=workspace_path, bo_steps=bo_steps)
        inner_cmd_seq = [suggest_cmd, hyperopt_cmd]
        if use_error_instructions:
            inner_cmd_seq = [error_instruct_cmd] + inner_cmd_seq
        inner_sequential_flow = SequentialFlow(
            sequence=inner_cmd_seq,
            name="suggest_search_space_and_run_hyperopt",
            description="suggest search space and run hyperopt",
        )
        loop_flow = LoopFlow(
            loop_body=inner_sequential_flow,
            max_repetitions=max_repetitions,
            allow_early_break=True,
            max_retries=max_retries,
            prompt_template=loop_flow_prompt_template,
            name=f"hyperopt_loop",
            parse_func_id="extract_action_as_json",
        )
        super().__init__(
            sequence=[summarize_cmd, loop_flow, Act()],
            name="hyperopt pipeline",
            description=f"{loop_flow.description}, then Summarize step and finally Act.",
        )


class SummarizeCode(HumanTakeoverCommand):
    name: str = "summarize_code"
    description: str = "write a summary of user provided code"
    required_prompt_templates: dict[str, str]
    input_keys: dict[str, MemKey] = {MemKey.CODE.value: MemKey.CODE}
    output_keys: dict[str, MemKey] = {MemKey.CODE_SUMMARY.value: MemKey.CODE_SUMMARY}
    max_retries: int = 5
    parse_func_id: str = "extract_summary_as_json"

    def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
        summary = safe_parsing_chat_completion(
            agent=agent,
            ask_template=self.required_prompt_templates["summarize_code_prompt_template"],
            prompt_kwargs={k: agent.memory.retrieve(self.input_keys[k]) for k in self.input_keys},
            parse_func=PARSE_FUNC_MAP[self.parse_func_id],
            format_error_message="Your response did no follow the required format"
            '\n```json\n{\n\t"summary": "<summary>"\n}\n```\n'
            "Correct it now.",
            max_retries=self.max_retries,
            human_takeover=self.check_trigger_human_takeover(),
        )
        agent.memory.store(content=summary, tags=self.output_keys[MemKey.CODE_SUMMARY.value])


class SuggestSearchSpace(HumanTakeoverCommand):
    name: str = "suggest_search_space"
    description: str = "suggest a search space for the hyperparameter optimization"
    required_prompt_templates: dict[str, str]
    input_keys: dict[str, MemKey] = {
        MemKey.CODE.value: MemKey.CODE,
        MemKey.CODE_SUMMARY.value: MemKey.CODE_SUMMARY,
    }
    output_keys: dict[str, MemKey] = {MemKey.BO_SEARCH_SPACE.value: MemKey.BO_SEARCH_SPACE}
    max_retries: int = 5

    def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
        search_space = safe_parsing_chat_completion(
            agent=agent,
            ask_template=self.required_prompt_templates["search_space_prompt_template"],
            prompt_kwargs={k: agent.memory.retrieve(self.input_keys[k]) for k in self.input_keys},
            parse_func=extract_json,
            format_error_message=(
                "Your response did no follow the required format\n"
                "```json\n"
                "{\n"
                "\t{'<model1_name>': [\n"
                "\t\t{'name': '<hyperparameter1_name>', 'type': '<type>', 'lb': <lower_bound>, 'ub': <upper_bound>, 'categories': [<categories>]},\n"
                "\t\t{'name': '<hyperparameter2_name>', 'type': '<type>', 'lb': <lower_bound>, 'ub': <upper_bound>, 'categories': [<categories>]},\n"
                " \t\tetc.\n"
                "\t],\n"
                "\t{'<model2_name>': [\n"
                "\t\t{'name': '<hyperparameter1_name>', 'type': '<type>', 'lb': <lower_bound>, 'ub': <upper_bound>, 'categories': [<categories>]},\n"
                "\t\t{'name': '<hyperparameter2_name>', 'type': '<type>', 'lb': <lower_bound>, 'ub': <upper_bound>, 'categories': [<categories>]},\n"
                " \t\tetc.\n"
                "\t],\n"
                "\tetc.\n"
                "}\n"
                "```\n"
                "Correct it now."
            ),
            max_retries=self.max_retries,
            human_takeover=self.check_trigger_human_takeover(),
        )
        agent.memory.store(content=search_space, tags=self.output_keys[MemKey.BO_SEARCH_SPACE.value])


class RunHyperOpt(HumanTakeoverCommand):
    name: str = "run_hyperopt"
    description: str = "run the main hyperopt loop"
    # required_prompt_templates: dict[str, str]
    input_keys: dict[str, MemKey] = {
        MemKey.CODE_SUMMARY.value: MemKey.CODE_SUMMARY,
        MemKey.BO_SEARCH_SPACE.value: MemKey.BO_SEARCH_SPACE,
    }
    output_keys: dict[str, MemKey] = {
        MemKey.BO_OBSERVATIONS.value: MemKey.BO_OBSERVATIONS,
        MemKey.BO_BEST_CANDIDATE.value: MemKey.BO_BEST_CANDIDATE,
        MemKey.BO_BEST_SCORE.value: MemKey.BO_BEST_SCORE,
        MemKey.BO_ERROR.value: MemKey.BO_ERROR,
        MemKey.BO_RAN.value: MemKey.BO_RAN,
    }
    workspace_path: str
    bo_steps: int
    timestamp: int = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    def __init__(self, **data: Any):
        super().__init__(**data)
        os.makedirs(os.path.join(self.workspace_path, "results", self.timestamp))

    @staticmethod
    def get_hebo_design_space(search_space: Dict[str, Dict[str, Any]]):
        search_spaces = []
        for model_name in search_space:
            model_name_str = model_name.lower() + "_"
            model_search_space = []
            for param_dict in search_space[model_name]:
                param_dict["name"] = model_name_str + param_dict["name"]
                model_search_space.append(param_dict)
            search_spaces += model_search_space
        space = DesignSpace().parse(search_spaces)
        return space

    def evaluate_candidate(self, params: pd.DataFrame) -> Tuple[Union[float, None], str]:
        # write code that calls the blackbox with the candidate parameters
        candidate_code_path = os.path.join(self.workspace_path, "code/candidate_code.py")
        candidate_code = "from blackbox import blackbox\n"
        params_str_list = dict()
        for p in params:
            if params[p].dtype == "object":
                params_str_list[p] = f'"{str(params[p].values[0])}"'
            else:
                params_str_list[p] = str(params[p].values[0])
        params_str = ", ".join([f"{k}={v}" for k, v in params_str_list.items()])
        candidate_code += f"score = blackbox({params_str})\n"
        candidate_code += f"print('SCORE:', score)"
        with open(candidate_code_path, "w") as f:
            f.write(candidate_code)

        # run new code and write results in file
        outfile = os.path.join(os.path.join(self.workspace_path, "code", "bo_output.txt"))
        errfile = os.path.join(os.path.join(self.workspace_path, "code", "bo_error.txt"))
        python_path = get_path_to_python("./third_party/hyperopt/path_to_python.txt")
        cmd = f"{python_path} {candidate_code_path} 2> {errfile} > {outfile}"
        os.system(cmd)

        with open(errfile) as f:
            error_str = f.read()
        if "Error" in error_str:
            print(error_str)
            return None, error_str

        with open(outfile) as f:
            output = f.readlines()
            out = output[-1]

        assert "SCORE" in out
        return float(out.split(":")[1]), out

    def run_optimization(self, search_space: Dict[str, Dict[str, Any]]) -> Tuple[HEBO, Union[str, None]]:
        design_space = self.get_hebo_design_space(search_space=search_space)
        optimizer = HEBO(design_space)

        for step_idx in range(self.bo_steps):
            # catch errors when retraining GP and suggesting a new candidate
            try:
                candidate = optimizer.suggest()
            except (ValueError, RuntimeError) as e:
                print(e)
                return optimizer, e

            y, error_str = self.evaluate_candidate(params=candidate)

            print("-" * 50)
            print(f"BO STEP #{step_idx}")
            print("-" * 50)
            print("candidate:\n", candidate.T)
            print("score:", y)
            print("output:", error_str)

            if y is None:
                return optimizer, error_str

            assert isinstance(y, float)
            y_use = -1.0 * np.array(y).reshape((-1, 1))  # hebo is minimizing
            optimizer.observe(candidate, y_use)

            if len(optimizer.X) > 0:
                self.save_optimization_trajectory(optimizer)

        return optimizer, None

    def create_blackbox(self, search_space: Dict[str, Dict[str, Any]]) -> None:
        """
        Creates a blackbox function by wrapping the code in a function exposing the parameters of the search space.
        """
        # read user code
        user_code_path = os.path.join(self.workspace_path, "code/executor_code.py")
        with open(user_code_path) as f:
            code = f.read()

        # replace parameters in user code with names from search space and write a function called `blackbox`
        # exposing these parameters
        blackbox_function_code = wrap_code(code=code, space=search_space)
        blackbox_code_path = os.path.join(self.workspace_path, "code/blackbox.py")
        with open(blackbox_code_path, "w") as f:
            f.write(blackbox_function_code)

    def func(self, agent: LLMAgent, *args, **kwargs) -> None:
        search_space = agent.memory.retrieve(self.input_keys[MemKey.BO_SEARCH_SPACE.value])
        self.create_blackbox(search_space=search_space)
        optimizer, error_str = self.run_optimization(search_space=search_space)

        if error_str is None:
            # insert the best hyperparameters in code and write it to new file
            user_code_path = os.path.join(self.workspace_path, "code/executor_code.py")
            with open(user_code_path) as f:
                code = "".join(f.readlines())
            optimized_code = assign_hyperopt(code=code, candidate=optimizer.best_x, space=search_space)
            updated_code_path = os.path.join(self.workspace_path, "results", self.timestamp, "optimized_code.py")
            with open(updated_code_path, "w") as f:
                f.write(optimized_code)

        if len(optimizer.X) > 0:
            observations = optimizer.X
            best_x = optimizer.best_x
            best_y = optimizer.best_y
            bo_ran = True
        else:
            observations = None
            best_x = None
            best_y = None
            bo_ran = False

        agent.memory.store(observations, MemKey.BO_OBSERVATIONS)
        agent.memory.store(best_x, MemKey.BO_BEST_CANDIDATE)
        agent.memory.store(best_y, MemKey.BO_BEST_SCORE)
        agent.memory.store(error_str, MemKey.BO_ERROR)
        agent.memory.store(bo_ran, MemKey.BO_RAN)

        # save optimization trajectory to workspace
        self.save_optimization_trajectory(optimizer)

    def save_optimization_trajectory(self, optimizer):
        trajectory = optimizer.X.copy()
        trajectory["y"] = -1.0 * optimizer.y
        trajectory.to_csv(os.path.join(self.workspace_path, "results", self.timestamp, "optimization_trajectory.csv"))


class ErrorInstruct(HumanTakeoverCommand):
    """
    Ask the Agent to interpret past errors and write a new instruction out of it in order to avoid doing the same
    mistake in the future. These are continuously extended until the unit test passes.
    """

    name: str = "error_instruct"
    description: str = "Transform past errors in new instructions"
    input_keys: dict[str, MemKey] = {
        MemKey.BO_RAN.value: MemKey.BO_RAN,
        MemKey.BO_ERROR.value: MemKey.BO_ERROR,
    }
    output_keys: dict[str, MemKey] = {MemKey.ERROR_INSTRUCT.value: MemKey.ERROR_INSTRUCT}
    max_retries: int = 5
    human_takeover: bool = False
    parse_func_id: str = "extract_instruction_as_json"

    def func(self, agent, *args: Any, **kwargs: Any):
        error_instructions = agent.memory.retrieve(MemKey.ERROR_INSTRUCT)  # cannot be in `input_keys` as it raises err
        bo_ran = agent.memory.retrieve(self.input_keys[MemKey.BO_RAN.value])
        bo_error = agent.memory.retrieve(self.input_keys[MemKey.BO_ERROR.value])

        if bo_ran and bo_error is not None and len(bo_error) > 0:
            response = safe_parsing_chat_completion(
                agent=agent,
                ask_template=self.required_prompt_templates["error_instruct_template"],
                parse_func=PARSE_FUNC_MAP[self.parse_func_id],
                format_error_message="Your response did no follow the required format"
                '\n```json\n{\n\t"instruction": "<instruction>"\n}\n```. Correct it now.',
                max_retries=self.max_retries,
                human_takeover=self.check_trigger_human_takeover(),
            )
            if len(response) > 0:
                if error_instructions is not None:
                    error_instructions += f"\n- {response}"
                else:
                    error_instructions = f"\n- {response}"
                agent.memory.store(error_instructions, self.output_keys[MemKey.ERROR_INSTRUCT.value])

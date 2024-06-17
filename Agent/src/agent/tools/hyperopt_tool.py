import ast
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

from agent.memory import MemKey
from agent.utils.hyperopt_utils import assign_hyperopt
from agent.utils.utils import get_path_to_python


class HyperOptTool:
    name: str = "HyperOpt tool"
    requires_llm_prompt: bool = True  # whether the tool needs llm output as input when it is called.

    def __init__(self, bo_steps: int, path_to_python: str, workspace_path: str):
        self.bo_steps = bo_steps
        self.workspace_path = workspace_path
        self.eval_model_path = os.path.join(self.workspace_path, "code/code.py")
        self.final_model_path = os.path.join(self.workspace_path, "code/best_code.py")
        self.out_path = os.path.join(self.workspace_path, "bo_accuracy.py")
        self.path_to_python = get_path_to_python(path_to_python)

    def __call__(self, agent_input: str) -> Dict[MemKey, str]:
        try:
            agent_input = agent_input.replace("\n\t", " ").replace("\n", " ")
            param_space_with_model = ast.literal_eval(re.search("({.+})", agent_input).group(0))
        except Exception:
            error = (
                "The agent does not generate hyperparameter space in the correct format. "
                "Please retry in the correct dictionary format."
            )
            return {MemKey.BO_ERROR: error}
        optimized_params, error = self.optimize(param_space_with_model=param_space_with_model)

        if error != "":
            return {MemKey.BO_ERROR: error}

        with open(self.eval_model_path) as f:
            model_code = "".join(f.readlines())
        final_model_code = assign_hyperopt(code=model_code, space=optimized_params)
        with open(self.final_model_path, "w") as f:
            f.write(final_model_code)

        cmd = f"{self.path_to_python} {self.final_model_path}"
        os.system(cmd + " > " + self.out_path)

        with open(self.out_path) as f:
            bo_performance = f.read()
        try:
            bo_final_performance = float(
                bo_performance[bo_performance.rfind("ACCURACY:") + 10: bo_performance.rfind("ACCURACY:") + 17]
            )
        except Exception:
            bo_final_performance = float(
                bo_performance[bo_performance.rfind("R2_SCORE:") + 10: bo_performance.rfind("R2_SCORE:") + 17]
            )

        return {MemKey.BO_PERF: bo_final_performance, MemKey.RAN_HYPEROPT: "True"}

    @staticmethod
    def get_design_space(param_space_with_model: Dict[str, Dict[str, List[Any]]]):
        params = []
        for func, func_params in param_space_with_model.items():
            for param_name, param_vals in func_params.items():
                params.append({"name": f"{func}__{param_name}", "type": "int", "lb": 0, "ub": len(param_vals) - 1})
        space = DesignSpace().parse(params)
        return space

    def pipeline_eval(self, params: Dict[str, Dict[str, Any]], path_to_code: str) -> Tuple[float, str]:
        """Evaluate the pipeline with the given hyperparameters.

        Args:
            params: dictionary (key: module name, value: dictionary mapping hyperparam name to their values)

        Returns:
            performance: metric to minimize
            error: error trace if an error happened
        """
        with open(path_to_code) as f:
            model_code = "".join(f.readlines())
        updated_model_code = assign_hyperopt(code=model_code, space=params)
        with open(path_to_code, "w") as f:
            f.write(updated_model_code)
        outfile = os.path.join(os.path.dirname(path_to_code), "bo_output.txt")
        cmd = f"{self.path_to_python} {path_to_code} > {outfile}"
        print(cmd)
        os.system(cmd)
        with open(outfile) as f:
            output = f.readlines()
            out = output[-1]
        if "ACCURACY" in out:
            performance = float(out.split(":")[1])
        elif "R2_SCORE" in out:
            performance = float(out.split(":")[1])
        else:
            return None, "".join(output)
        return -performance, ""

    def optimize(self, param_space_with_model: Dict[str, Dict[str, List[Any]]]) -> Tuple[Dict[str, Any], str]:
        space = self.get_design_space(param_space_with_model=param_space_with_model)
        bo = HEBO(space)
        for _ in range(self.bo_steps):
            rec_x = bo.suggest()
            perfs = []
            for params in self.format_f_inputs(x=rec_x, param_space_with_model=param_space_with_model):
                perf, error = self.pipeline_eval(params=params, path_to_code=self.eval_model_path)
                if error == "":
                    perfs.append(perf)
                else:
                    return None, error
            bo.observe(rec_x, np.array(perfs).reshape((-1, 1)))
        optimized = self.format_f_inputs(x=bo.best_x, param_space_with_model=param_space_with_model)[0]
        return optimized, ""

    @staticmethod
    def format_f_inputs(
            x: pd.DataFrame, param_space_with_model: Dict[str, Dict[str, List[Any]]]
    ) -> List[Dict[str, Any]]:
        formatted_inputs = []
        for i in range(len(x)):
            new_input = {}
            for function_param_name, ind in x.iloc[i].to_dict().items():
                function, param_name = function_param_name.split("__")
                if function not in new_input:
                    new_input[function] = {}
                new_input[function][param_name] = param_space_with_model[function][param_name][ind]
            formatted_inputs.append(new_input)
        return formatted_inputs

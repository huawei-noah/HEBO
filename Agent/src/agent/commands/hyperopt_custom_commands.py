import copy
import os.path
from typing import Union, Any, Tuple, Optional

import numpy as np
import pandas as pd
from hebo.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

from agent.agents import LLMAgent, safe_parsing_chat_completion
from agent.commands import SequentialFlow, LoopFlow, Act
from agent.commands.core import HumanTakeoverCommand
from agent.commands.flows import PARSE_FUNC_MAP
from agent.memory import MemKey
from agent.utils.hyperopt_utils import HYPEROPT_FORMAT_ERROR_MESSAGE
from agent.utils.hyperopt_utils import assign_hyperopt, wrap_code, unwrap_code
from agent.utils.utils import get_path_to_python, extract_json


class HyperOpt(SequentialFlow):
    def __init__(
            self,
            summarize_code_prompt_template: str,
            k_folds_cv_prompt_template: str,
            loop_flow_prompt_template: str,
            error_instruct_prompt_template: str,
            search_space_prompt_template: str,
            reflection_strategy: str | None,
            bo_steps: int,
            bo_batch_size: int = 1,
            bo_search_spaces: int = 1,
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
        if reflection_strategy is None:
            bo_search_spaces = 1
        if reflection_strategy is not None:
            bo_steps = bo_steps // bo_search_spaces
        print(
            f"Doing {bo_search_spaces * bo_steps} BO steps: "
            f"{bo_search_spaces} search spaces with {bo_steps} steps each "
            f"(note max_repetitions={max_repetitions})",
            flush=True
        )

        summarize_cmd = SummarizeCode(
            required_prompt_templates={"summarize_code_prompt_template": summarize_code_prompt_template},
            max_retries=max_retries
        )
        k_folds_cv_cmd = KFoldsCV(
            required_prompt_templates={"k_folds_cv_prompt_template": k_folds_cv_prompt_template},
            max_retries=max_retries
        )

        error_instruct_cmd = ErrorInstruct(
            required_prompt_templates={"error_instruct_template": error_instruct_prompt_template},
            max_retries=max_retries
        )
        suggest_cmd = SuggestSearchSpace(
            required_prompt_templates={"search_space_prompt_template": search_space_prompt_template},
            max_retries=max_retries
        )
        hyperopt_cmd = RunHyperOpt(
            reflection_strategy=reflection_strategy,
            bo_steps=bo_steps,
            bo_batch_size=bo_batch_size,
            search_space_limit=bo_search_spaces,
        )
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
            parse_func_id='extract_action_as_json',
            memory_choice_tag_val=MemKey.CONTINUE_OR_TERMINATE_BO,
        )
        super().__init__(
            sequence=[summarize_cmd, k_folds_cv_cmd, loop_flow, Act()],
            name='hyperopt pipeline',
            description=f"{loop_flow.description}, then Summarize step and finally Act."
        )


class SummarizeCode(HumanTakeoverCommand):
    name: str = "summarize_code"
    description: str = "write a summary of user provided code"
    required_prompt_templates: dict[str, str]
    input_keys: dict[str, MemKey] = {MemKey.CODE.value: MemKey.CODE}
    output_keys: dict[str, MemKey] = {MemKey.CODE_SUMMARY.value: MemKey.CODE_SUMMARY}
    max_retries: int = 5
    parse_func_id: str = 'extract_summary_as_json'

    def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
        summary = safe_parsing_chat_completion(
            agent=agent,
            ask_template=self.required_prompt_templates["summarize_code_prompt_template"],
            prompt_kwargs={k: agent.memory.retrieve(self.input_keys[k]) for k in self.input_keys},
            parse_func=PARSE_FUNC_MAP[self.parse_func_id],
            format_error_message='Your response did not follow the required format'
                                 '\n```json\n{\n\t"summary": "<summary>"\n}\n```\n'
                                 'Correct it now.',
            max_retries=self.max_retries,
            human_takeover=self.check_trigger_human_takeover(),
        )
        agent.memory.store(content=summary, tags=self.output_keys[MemKey.CODE_SUMMARY.value])


class KFoldsCV(HumanTakeoverCommand):
    name: str = "kfolds_cv"
    description: str = "run kfold cross validation"
    required_prompt_templates: dict[str, str]
    input_keys: dict[str, MemKey] = {
        MemKey.CODE.value: MemKey.CODE,
        MemKey.CODE_SUMMARY.value: MemKey.CODE_SUMMARY
    }
    output_keys: dict[str, MemKey] = {MemKey.K_FOLD_CV.value: MemKey.K_FOLD_CV}
    max_retries: int = 5

    def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
        folds = safe_parsing_chat_completion(
            agent=agent,
            ask_template=self.required_prompt_templates["k_folds_cv_prompt_template"],
            prompt_kwargs={k: agent.memory.retrieve(self.input_keys[k]) for k in self.input_keys},
            parse_func=extract_json,
            format_error_message=(
                "Your response did not follow the required format\n"
                "```json\n"
                "{\n"
                "\t'model': <ml model object or instance in user code>\n"
                "\t'X': <The final processed feature data in user code>\n"
                "\t'y': <the target labels data name in user code>\n"
                "\t'metric_func': <the evaluation metric function in user code>\n"
                "\t'metric_value_direction': <choose between MAX and MIN, when MAX, "
                "the metric value higher means model performance better, otherwise MIN>\n"
                "}\n"
                "```\n"
                "Correct it now."
            ),
            max_retries=self.max_retries,
            human_takeover=self.check_trigger_human_takeover(),
        )
        agent.memory.store(content=folds, tags=self.output_keys[MemKey.K_FOLD_CV.value])


class SuggestSearchSpace(HumanTakeoverCommand):
    name: str = "suggest_search_space"
    description: str = "suggest a search space for the hyperparameter optimization"
    required_prompt_templates: dict[str, str]
    input_keys: dict[str, MemKey] = {
        MemKey.CODE.value: MemKey.CODE,
        MemKey.CODE_SUMMARY.value: MemKey.CODE_SUMMARY,
        MemKey.BO_BEST_SCORE.value: MemKey.BO_BEST_SCORE,
        MemKey.BO_BEST_CANDIDATE.value: MemKey.BO_BEST_CANDIDATE,
        MemKey.BO_BEST_SEARCH_SPACE.value: MemKey.BO_BEST_SEARCH_SPACE,
        MemKey.BO_HISTORY.value: MemKey.BO_HISTORY,
    }
    output_keys: dict[str, MemKey] = {MemKey.BO_SEARCH_SPACE.value: MemKey.BO_SEARCH_SPACE}
    max_retries: int = 5

    @staticmethod
    def format_bo_history(bo_history: dict[str, dict[str, Any]]) -> str | None:
        formatted_bo_history_str = []
        if bo_history is None:
            return None
        for search_space_name in bo_history:
            best_x = bo_history[search_space_name]['best_x']
            best_y = bo_history[search_space_name]['best_y']
            search_space = bo_history[search_space_name]['search_space']
            _formatted_str = (f'On {search_space_name}\n{search_space}\n\n'
                              f'The best hyperparameter setting found was\n{best_x}\n\n'
                              f'With corresponding score {best_y}')
            formatted_bo_history_str.append(_formatted_str)
        return '-----\n'.join(formatted_bo_history_str) + '\n-----'

    def func(self, agent: LLMAgent, *args: Any, **kwargs: Any):
        formatted_bo_history = self.format_bo_history(agent.memory.retrieve(self.input_keys[MemKey.BO_HISTORY.value]))
        search_space = safe_parsing_chat_completion(
            agent=agent,
            ask_template=self.required_prompt_templates["search_space_prompt_template"],
            prompt_kwargs={'formatted_bo_history': formatted_bo_history},
            parse_func=extract_json,
            format_error_message=HYPEROPT_FORMAT_ERROR_MESSAGE,
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
        MemKey.K_FOLD_CV.value: MemKey.K_FOLD_CV,
        MemKey.RESULTS_DIR.value: MemKey.RESULTS_DIR,
        MemKey.CODE_DIR.value: MemKey.CODE_DIR,
    }
    output_keys: dict[str, MemKey] = {
        MemKey.BO_OBSERVATIONS.value: MemKey.BO_OBSERVATIONS,
        MemKey.BO_BEST_CANDIDATE.value: MemKey.BO_BEST_CANDIDATE,
        MemKey.BO_BEST_SCORE.value: MemKey.BO_BEST_SCORE,
        MemKey.BO_BEST_SEARCH_SPACE.value: MemKey.BO_BEST_SEARCH_SPACE,
        MemKey.BO_HISTORY.value: MemKey.BO_HISTORY,
        MemKey.BO_ERROR.value: MemKey.BO_ERROR,
        MemKey.BO_RAN.value: MemKey.BO_RAN,
        MemKey.CONTINUE_OR_TERMINATE_BO.value: MemKey.CONTINUE_OR_TERMINATE_BO,
    }
    bo_steps: int
    bo_batch_size: int
    search_space_limit: int
    results_path: str | None = None
    code_dir: str | None = None
    reflection_strategy: str | None = None
    search_space_counter: int = 0

    @staticmethod
    def get_hebo_design_space(search_space: dict[str, dict[str, Any]]):
        search_space_ = copy.deepcopy(search_space)
        search_spaces = []
        for model_name in search_space_:
            model_name_str = ''.join(model_name.lower().split()) + '_'
            model_search_space = []
            for param_dict in search_space_[model_name]:
                param_dict['name'] = model_name_str + param_dict['name']
                model_search_space.append(param_dict)
            search_spaces += model_search_space
        space = DesignSpace().parse(search_spaces)
        return space

    def evaluate_candidate(self, params: pd.DataFrame) -> Tuple[Union[float, None], str]:
        # write code that calls the blackbox with the candidate parameters
        candidate_code_path = os.path.join(self.code_dir, 'candidate_code.py')
        candidate_code = "from blackbox import blackbox\n"
        params_str_list = dict()
        for p in params:
            if params[p].dtype == 'object':
                if params[p].values[0] in [None, True, False]:
                    params_str_list[p] = params[p].values[0]
                else:
                    params_str_list[p] = f'\"{str(params[p].values[0])}\"'
            else:
                params_str_list[p] = str(params[p].values[0])
        params_str = ", ".join([f'{k}={v}' for k, v in params_str_list.items()])
        candidate_code += f"score = blackbox({params_str})\n"
        candidate_code += f"print('SCORE:', score)"
        with open(candidate_code_path, 'w') as f:
            f.write(candidate_code)

        # run new code and write results in file
        outfile = os.path.join(os.path.join(self.code_dir, "bo_output.txt"))
        errfile = os.path.join(os.path.join(self.code_dir, "bo_error.txt"))
        python_path = get_path_to_python('./third_party/hyperopt/path_to_python.txt')
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

    def run_optimization(
            self,
            search_space: dict[str, dict[str, Any]],
            initial_points: Optional[Tuple[pd.DataFrame, np.ndarray]] = None,
    ) -> Tuple[HEBO, Union[str, None]]:
        design_space = self.get_hebo_design_space(search_space=search_space)
        optimizer = HEBO(design_space)

        if initial_points is not None:
            X_init, y_init = initial_points
            optimizer.observe(X_init, y_init)

        for step_idx in range(self.bo_steps):
            # catch errors when retraining GP and suggesting a new candidate
            try:
                candidate = optimizer.suggest(n_suggestions=self.bo_batch_size)
            except (ValueError, RuntimeError) as e:
                print(e)
                return optimizer, e

            y, error_str = self.evaluate_candidate(params=candidate)

            print('-' * 50)
            print(f'BO STEP #{step_idx}')
            print('-' * 50)
            print('candidate:\n', candidate.T)
            print('score:', y)
            print('output:', error_str)

            if y is None:
                return optimizer, error_str

            assert isinstance(y, float)
            optimizer.observe(candidate, np.array(y).reshape((-1, 1)))

        return optimizer, None

    def create_blackbox(self, search_space: dict[str, dict[str, Any]], cv_args: dict[str, Any]) -> None:
        """
        Creates a blackbox function by wrapping the code in a function exposing the parameters of the search space.
        """
        # read user code
        user_code_path = os.path.join(self.code_dir, 'code.py')
        with open(user_code_path) as f:
            code = f.read()

        # replace parameters in user code with names from search space and write a function called `blackbox`
        # exposing these parameters
        blackbox_code_path = os.path.join(self.code_dir, 'blackbox.py')
        blackbox_function_code = wrap_code(code=code, space=search_space, cv_args=cv_args)
        with open(blackbox_code_path, 'w') as f:
            f.write(blackbox_function_code)

    @staticmethod
    def observation_in_search_space(observation: pd.Series, design_space: DesignSpace) -> bool:
        for name in observation.index:
            if name in design_space.numeric_names:
                if not design_space.paras[name].lb <= observation[name] <= design_space.paras[name].ub:
                    return False
                if design_space.paras[name].is_discrete and int(observation[name]) != observation[name]:
                    return False
            elif name in design_space.enum_names:
                if not observation[name] in design_space.paras[name].categories:
                    return False
        return True

    def get_reusable_observations(
            self, search_space: dict[str, dict[str, Any]]
    ) -> Union[Tuple[pd.DataFrame, np.ndarray], None]:
        """
        Reads past trajectories and checks all past observations for ones that fall in the new current search space,
        so we can reuse them in the GP model.

        :param search_space: dictionary of search space per hyperparameter
        :return: X observations (pd.DataFrame), y observations (np.ndarray)
        """
        if self.search_space_counter == 0:
            return None

        design_space = self.get_hebo_design_space(search_space=search_space)
        reused_obs = pd.DataFrame()
        for i in range(self.search_space_counter):
            traj_i = pd.read_csv(
                os.path.join(self.results_path, f'search_space_{i}', 'optimization_trajectory.csv'))
            X_i = traj_i.loc[:, traj_i.columns != 'y']
            for k, obs_k in X_i.iterrows():
                if self.observation_in_search_space(observation=obs_k, design_space=design_space):
                    reused_obs = reused_obs._append(traj_i.iloc[[k]], ignore_index=True)

        if len(reused_obs) == 0:
            return None
        return reused_obs.loc[:, reused_obs.columns != 'y'], reused_obs['y'].values

    def func(self, agent: LLMAgent, *args, **kwargs) -> None:
        self.results_path = agent.memory.retrieve(self.input_keys[MemKey.RESULTS_DIR.value])
        self.code_dir = agent.memory.retrieve(self.input_keys[MemKey.CODE_DIR.value])
        os.makedirs(
            os.path.join(self.results_path, f'search_space_{self.search_space_counter}'), exist_ok=True
        )
        search_space = agent.memory.retrieve(self.input_keys[MemKey.BO_SEARCH_SPACE.value])
        cv_args = copy.deepcopy(agent.memory.retrieve(self.input_keys[MemKey.K_FOLD_CV.value]))
        self.create_blackbox(search_space=search_space, cv_args=cv_args)
        reusable_observations = self.get_reusable_observations(search_space=search_space)
        optimizer, error_str = self.run_optimization(search_space=search_space, initial_points=reusable_observations)

        if error_str is None:
            # insert the best hyperparameters in code and write it to new file
            top5_indices = np.argsort(optimizer.y.flatten())[:5]
            top5_configs = optimizer.X.iloc[top5_indices]
            for i, row in top5_configs.iterrows():
                blackbox_code_path = os.path.join(self.code_dir, 'blackbox.py')
                with open(blackbox_code_path, 'r') as f:
                    blackbox_code_lines = f.readlines()
                code = unwrap_code(blackbox_code_lines)
                code = assign_hyperopt(code=code, candidate=row, space=search_space)
                updated_code_path = os.path.join(
                    self.results_path, f'search_space_{self.search_space_counter}', f'optimized_code_{i}.py'
                )
                with open(updated_code_path, "w") as f:
                    f.write(code)

        if len(optimizer.X) > 0 and error_str is None:
            n_init = len(reusable_observations[0]) if reusable_observations is not None else 0
            observations = optimizer.X.iloc[n_init:]
            best_x = optimizer.best_x
            best_y = optimizer.best_y
            bo_ran = True

            # save optimization trajectory to workspace
            trajectory = optimizer.X.iloc[n_init:]
            trajectory['y'] = optimizer.y[n_init:]
            trajectory.to_csv(os.path.join(
                self.results_path, f'search_space_{self.search_space_counter}', 'optimization_trajectory.csv'
            ), index=False)

            # save overall best candidate and its corresponding score and search space (across all search spaces seen)
            best_score = agent.memory.retrieve(MemKey.BO_BEST_SCORE)
            if best_score is None or best_y < best_score:
                print("New overall best score found!", flush=True)
                agent.memory.store(best_y, MemKey.BO_BEST_SCORE)
                agent.memory.store(best_x, MemKey.BO_BEST_CANDIDATE)
                agent.memory.store(search_space, MemKey.BO_BEST_SEARCH_SPACE)

            # append to history, current search space, current best candidate and current best value
            bo_history = agent.memory.retrieve(MemKey.BO_HISTORY)
            if bo_history is None:
                bo_history = {}
            bo_history[f'search_space_{self.search_space_counter}'] = {
                'best_x': best_x, 'best_y': best_y, 'search_space': search_space
            }
            agent.memory.store(bo_history, MemKey.BO_HISTORY)

            self.search_space_counter += 1

        else:
            observations = None
            bo_ran = False

        agent.memory.store(observations, MemKey.BO_OBSERVATIONS)
        agent.memory.store(error_str, MemKey.BO_ERROR)
        agent.memory.store(bo_ran, MemKey.BO_RAN)

        if self.search_space_counter >= self.search_space_limit:
            agent.memory.store("Terminate", MemKey.CONTINUE_OR_TERMINATE_BO)


class ErrorInstruct(HumanTakeoverCommand):
    """
    Ask the Agent to interpret past errors and write a new instruction out of it in order to avoid doing the same
    mistake in the future. These are continuously extended until the unit test passes.
    """
    name: str = 'error_instruct'
    description: str = 'Transform past errors in new instructions'
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
                ask_template=self.required_prompt_templates['error_instruct_template'],
                parse_func=PARSE_FUNC_MAP[self.parse_func_id],
                format_error_message='Your response did not follow the required format'
                                     '\n```json\n{\n\t"instruction": "<instruction>"\n}\n```. Correct it now.',
                max_retries=self.max_retries,
                human_takeover=self.check_trigger_human_takeover()
            )
            if len(response) > 0:
                if error_instructions is not None:
                    error_instructions += f'\n- {response}'
                else:
                    error_instructions = f"\n- {response}"
                agent.memory.store(error_instructions, self.output_keys[MemKey.ERROR_INSTRUCT.value])

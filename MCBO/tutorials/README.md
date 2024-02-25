# Guide to use and extend MCBO

- [Optimize a custom black-box](#custom-black-box)
- [Adding new BO modules](#adding-new-bo-modules)
    - [Adding a surrogate model](#adding-a-surrogate-model)
    - [Adding an acquisition function](#adding-an-acquisition-function)
    - [Adding an acquisition function optimizer](#adding-an-acquisition-function-optimizer)
- [Evaluate a new BO algorithm on the MCBO tasks](#evaluate-a-new-bo-algorithm-on-the-mcbo-tasks)
- [Runtime estimation](#runtime-estimation)

---

## Custom black-box

This section shows how to run MCBO optimizers on tasks that are not already supported by the library.

Whether you want to include the black-box to the MCBO library or not, you need to define a new task
inheriting [TaskBase](../mcbo/tasks/task_base.py)

For illustrative purpose, consider a simple function $f(x)$ defined on a mixed space.
The first 3 elements of the search space (named `op0`, `op1`, `op2`) are categorical, with categories being the
operators `cosinus`, `sinus` and `exponential`, and the 3 last elements are numerical elements (`x0`, `x1`, `x2`) taking
values in interval [-1, 1], and the objective function is $f(x) = \text{op0}(\text{x0}) / (1 + \text{op1}(\text{x1})) +
\text{op2}(\text{op2})$.

```python
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from mcbo.tasks.task_base import TaskBase


class CustomTask(TaskBase):
    op_converter = {'cos': np.cos, 'sin': np.sin, 'exp': np.exp}

    @property
    def name(self) -> str:
        return 'Custom Task'

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        y = np.zeros((len(x), 1))  # will be filled with evaluations
        for ind in range(len(x)):
            x_ind = x.iloc[ind].to_dict()  # convert to a dictionary
            ops = [self.op_converter[x_ind[f'op{j}']] for j in range(3)]
            y[ind] = ops[0](x_ind['x0']) / (1 + ops[1](x_ind['x1'])) + ops[2](x_ind['x2'])
        return y

    def get_search_space_params(self) -> List[Dict[str, Any]]:
        params = [{'name': f'op{i}', 'type': 'nominal', 'categories': ['cos', 'sin', 'exp']} for i in range(3)]
        params.extend([{'name': f'x{i}', 'type': 'num', 'lb': -1, 'ub': 1} for i in range(3)])
        return params
```

Check out [tuto_custom_task.ipynb](./tuto_custom_task.ipynb) for an example of a custom task with input constraints and
with a full optimization loop.

**What if the custom black-box requires calling functions or running experiments outside of python?**

- In that case, it is possible to
  call `os.system(f"Your command with some args -x {x[0]}... > out.log")` in the `evaluate` method of the custom task
  and to parse the output in `out.log` using `with open("out.log", "r") as f:...` to get the output. You can
  check [MIG synthesis flow task](../mcbo/tasks/mig_seq_opt/mig_seq_opt_task.py) which uses this technique.
- Moreover, for an even more general case where the evaluation of the black-box requires conducting real-world
  experimentations (molecule design...), the `evaluate` method can simply make a
  call to `y[i] = input(f"Enter manually the black-box value associated to input {x.iloc[i]})` and let the experimenter
  to fill the black-box value associated to each entry of the suggested `x`.

##### Integrating the new task to the library

- Create a folder containing the code of the new task in the [tasks folder](../mcbo/tasks/).
- The folder can also contain a README providing a description of the task, associated search_space, dependencies...
- Add the new task to the [factory script](../mcbo/task_factory.py) (prefer local import of the new task class if it
  depends on packages not listed in [requirements.txt](../requirements.txt)). Also consider adding the task
  to [get_task_from_id](../mcbo/utils/experiment_utils.py) and associate it to a specific `task_id` (see examples
  of `xgboost_opt`, `antibody_design`, `pest`) to be able to run optimization on this task from command
  line, e.g. `./experiments/run_task_exps.py --task_id $task_id --optimizers_ids gp_to__ga__ei__basic --seeds 42 43 44`
- If you want to share the new task with the community, add reference to the task in the root [README.md](../README.md)
  in the dedicated section, and make a pull request.

---

## Adding new BO modules

This section describes the high-level procedure to integrate new modules (surrogate model, acquisition function,
acquisition function optimizer, and trust-region) to our MCBO framework. To allow mix-and-match via the `BoBuilder`.

### Adding a surrogate model

###### 1. Model implementation

The new model should be implemented in `/mcbo/models/` directory and inherit from
the [ModelBase](../mcbo/models/model_base.py)
class (or one of its subclasses, such as [ExactGPModel](../mcbo/models/gp/exact_gp.py)).
The new model should implement the following methods:

- `name`: just the name of the model used for logging
- `y_to_fit_y`: transform of the raw `ys` in black-box output domain (e.g. normalize / standardize them) to an adjusted
  domain.
- `fit_y_to_y`: map back the elements of the transformed domain to the original domain.
- `fit`: fit the model (fit the model's parameters).
- `predict`: return the model prediction as a mean and a variance (in original black-box output space, not in the
  transformed space).
- `noise`: return estimated noise variance in original domain.
- `to`: convert model to target device and dtype (pytorch related).
- (Optionally) `pre_fit_method`: this method is called before fitting the model in the suggest method. Can be used to
  update the internal state of the model. Use cases may include training a VAE for latent space BO, or re-initialising
  the model before fitting it to the data.

**Remark**: if the model is an exact GP with a new kernel, simply implement the kernel
in [kernel.py](../mcbo/models/gp/kernels.py), associate it to a kernel name and make it accessible in
the [kernel_factory](../mcbo/models/gp/kernel_factory.py) function (see other kernels s.a. `diffusion`, `rbf`, etc.).

###### 2. Add a test script to check fit and predict

- Add a test script in [/tests/models/](../tests/models/). You can
  adapt [test_exact_gp](../tests/models/test_exact_gp.py) replacing the exact GP with the new model.

**Remark**: if the model is an exact GP with a new kernel, also add a script in the same folder to check the kernel
itself.

###### 3. Add model to the BoBuilder to allow mix-and-match

- Allow the selection of the new model as a module of a MCBO algorithm by choosing a model id (
  see `gp_to`, `lr_sparse_hs`, etc.) and adding it in the `get_model` method
  of [BoBuilder class](../mcbo/optimizers/bo_builder.py). You can also add a dictionary of default
  hyperparameters `DEFAULT_MODEL_{MODEL_ID}_KWARGS` at the beginning of the same file (see for
  instance `DEFAULT_MODEL_EXACT_GP_KWARGS`).
- Then the model can be chosen as a module of a MCBO algorithm via the `BoBuilder`.

```python
from mcbo.optimizers.bo_builder import BoBuilder

new_model_id = "new_model_id"  # ID of the new model

task = ...
sp = task.get_search_space()
bo_builder = BoBuilder(model_id=new_model_id, acq_opt_id=..., acq_func_id=..., tr_id=...).build_bo(sp, n_init=20)
```

### Adding an acquisition function

###### 1. Acquisition function implementation

The new acquisition function should be implemented in `/mcbo/acq_funcs/` directory inherit
from [AcqBase](../mcbo/models/model_base.py) class (or [SingleObjAcqBase](../mcbo/models/model_base.py) for single
objective unconstrained acquisition functions such as [EI](../mcbo/acq_funcs/ei.py)).
The new acquisition function should implement the following methods:

- `name`: just the name of the acquisition function used for logging.
- `num_obj`: number of acquisition objectives.
- `num_constr`: number of black-box constraints (set to 0 for SingleObjAcqBase).
- `evaluate`: return acquisition values of all the points given as inputs.

**Remark**: as we assume acquisition function optimizers MINIMIZE acquisition functions, the `evaluate` method should
return a value to be MINIMIZED (e.g. [EI](../mcbo/acq_funcs/ei.py) actually returns the OPPOSITE of the standard
expected improvement).

###### 2. Add the acquisition function to the factory

- Choose a string id for the new acquisition function
- Allow selection of the new acquisition function via its id by adding it to
  the [acq_factory](../mcbo/acq_funcs/factory.py) function.

Doing this, you can instantiate the new acquisition function simply from its string id.

```python
from mcbo.acq_funcs.factory import acq_factory

acq_func = acq_factory(acq_func_id='new_acq_id', **kwargs)
```

###### 3. Add a test script

- Add a test script in [/tests/acq/](../tests/acq/). You can
  adapt [test_ei_acq](../tests/acq/test_ei_acq.py) replacing the expected improvement string id `'ei'` by the id of the
  new acquisition function.

###### 4. Add acquisition function to the BoBuilder to allow mix-and-match

- If the acquisition function does not require specific kwargs then nothing needs to be done since the BoBuilder will
  simply call the [acq_factory](../mcbo/acq_funcs/factory.py) to get the acquisition function.
- If during BO, the acquisition function needs run-specific parameters beyond input points and surrogate model (e.g. the
  best value observed so far for Expected improvement), then you need to modify the method `method_suggest`
  from [BoBase](../mcbo/optimizers/bo_base.py) to add the extra kwargs in
  the `acq_evaluate_kwargs` dictionary (e.g. `'best_y': best_y` for EI).

```python
from mcbo.optimizers.bo_builder import BoBuilder

new_acq_func_id = "new_acq_func_id"  # ID of the new model

task = ...
sp = task.get_search_space()
bo_builder = BoBuilder(model_id=..., acq_opt_id=..., acq_func_id=new_acq_func_id, tr_id=...).build_bo(sp, n_init=20)
```

### Adding an acquisition function optimizer

###### 1. Acquisition function optimizer implementation

The new acquisition function optimizer should be implemented in `/mcbo/acq_optimizers/` directory and inherit from
the [AcqOptimizerBase](../mcbo/acq_optimizers/acq_optimizer_base.py) class.
The new model should implement the following methods:

- `name`: just the name of the acquisition function optimizer used for logging.
- `optimize`: the core method of the class. It should return `n_suggestions` points to evaluate next. It notably takes
  as input a starting point, a fitted surrogate model, an acquisition function, the set of already observed points, and
  a trust region manager. If possible, it should be able to cope with simple input constraints (i.e. constraints that
  are cheap to check), for instance via rejection sampling. Input constraints are available via the `input_constraints`
  attribute of the instance.

**Remark 1**: the `optimize` method of the acquisition function optimizer must be seen as the method designed to
suggest the next points to evaluate. So if more than one suggestions are needed, it is possible to make the `optimize`
method call an internal `optimize_` method that would suggest a single point, and to
call `add_hallucinations_and_retrain_model` between two suggestions to refit the model without calling the black-box (as
done in [interleaved_search_acq_optimizer.py](../mcbo/acq_optimizers/interleaved_search_acq_optimizer.py)).

**Remark 2**: acquisition function optimizers be built as a wrapper of a non-BO optimizer that can be included in the
MCBO library under [/mcbo/optimizers/non_bo/](../mcbo/optimizers/non_bo). Therefore, a solution to implement an
acquisition function optimizer can be to start by implementing a non-BO optimizer (that support optimization in a fixed
trust region), and to use it in the new acquisition function optimizer. Examples of such acquisition function optimizers
are [GeneticAlgoAcqOptimizer](../mcbo/acq_optimizers/genetic_algorithm_acq_optimizer.py)
wrapping [GeneticAlgorithm](../mcbo/optimizers/non_bo/genetic_algorithm.py),
[MixedMabAcqOptimizer](../mcbo/acq_optimizers/mixed_mab_acq_optimizer.py)
wrapping [MultiArmedBandit](../mcbo.optimizers.non_bo.multi_armed_bandit.py),
or [RandomSeachAcqOptimizer](../mcbo/acq_optimizers/random_search_acq_optimizer.py)
wrapping [RandomSearch](../mcbo/optimizers/non_bo/random_search.py).

###### 2. Add a test script to assess the new acquisition function optimizer

- Add a test script in [/tests/acq_optimizers/](../tests/acq_optimizers/). You can
  adapt [test_rs_acq_opt.py](../mcbo/acq_optimizers/test_rs_acq_opt.py) replacing the `RandomSearchAcqOptimizer` with
  the new acquisition function optimizer.

###### 3. Add acquisition function optimizer to the BoBuilder to allow mix-and-match

- Allow the selection of the new acquisition function optimizer as a module of a MCBO algorithm by choosing a model id (
  see `ls`, `sa`, `ga`, etc.) and adding it in the `get_acq_optim` method
  of [BoBuilder class](../mcbo/optimizers/bo_builder.py). You can also add a dictionary of default
  hyperparameters `DEFAULT_ACQ_OPTIM_{ACQ_OPT_ID}_KWARGS` at the beginning of the same file (see for
  instance `DEFAULT_ACQ_OPTIM_IS_KWARGS` given for interleaved optimization).
- Then the acquisition function optimizer can be chosen as a module of a MCBO algorithm via the `BoBuilder`.

```python
from mcbo.optimizers.bo_builder import BoBuilder

new_acq_opt_id = "new_acq_opt_id"  # ID of the new model

task = ...
sp = task.get_search_space()
bo_builder = BoBuilder(model_id=..., acq_opt_id=new_acq_opt_id, acq_func_id=..., tr_id=...).build_bo(sp, n_init=20)
```

--- 

## Evaluate a new BO algorithm on the MCBO tasks

Once the new module is integrated into the `BoBuilder`, it is possible to evaluate a new mix-and-match MCBO algorithm on
the library tasks by simply calling [run_task_exps.py](../experiments/run_task_exps.py) (which is a wrapper around
the [run_experiment](../mcbo/utils/experiment_utils.py) function that can be checked if finer-grain control is needed).

We provide an example where a user has implemented a new model whose id in `BoBuilder` is `"new_model"` and wants to
assess its performance with GA and SA acquisition function optimization, and with and without using a basic
trust-region.
The following bash scripts can be adapted to exploit available cores and
GPUs (e.g. by parallelizing over one of the loops by adding `&` and `wait` and changing the `--device_id` so that each
command
runs on a different GPU).

- For combinatorial tasks

```shell
# Should be launched at the root of the project
model=NEW_MODEL_ID
acq_func="ei"
init_sampling_strategy="uniform"  # can be "sobol" or "sobol_scramble" to use initial sobol samples

SEEDS="42 43 44 45 46 47 48 49 50 51" 
ABSOLUT_EXE="./libs/Absolut/src/AbsolutNoLib"
for task in ackley aig_optimization antibody_design mig_optimization pest rna_inverse_fold; do
  for acq_opt in ga sa; do
    for tr in "basic" "none"; do  # whether to use a basic TR or not 
      opt_id="${model}__${acq_opt}__${acq_func}__${tr}__${init_sampling_strategy}"
      cmd="python ./experiments/run_task_exps.py --device_id 0 --absolut_dir $ABSOLUT_EXE --task_id $task \
                  --optimizers_ids $opt_id --seeds $SEEDS"
      echo $cmd
      $cmd
    done
  done
done

```

- For mixed-variable tasks

```shell
# To launch at # Should be launched at the root of the project
model=NEW_MODEL_ID
acq_func="ei"
init_sampling_strategy="uniform"  # can be "sobol" or "sobol_scramble" to use initial sobol samples

for task in ackley-53 xgboost_opt aig_optimization_hyp svm_opt; do
  for acq_opt in ga sa; do
    for tr in "basic" "none"; do  # whether to use a basic TR or not 
      opt_id="${model}__${acq_opt}__${acq_func}__${tr}__${init_sampling_strategy}"
      cmd="python ./experiments/run_task_exps.py --device_id 0 --absolut_dir $ABSOLUT_EXE --task_id $task \
                  --optimizers_ids $opt_id --seeds $SEEDS"
      echo $cmd
      $cmd
    done
  done
done

```

We provide the notebook [tuto_results_viz.ipynb](./tuto_results_viz.ipynb) to show how to load and visualize the
results, as well as how to compare to the baselines we
have run and for which we provide the [results](../paper_results/).

###### Black-box evaluation runtime

We provide a table of runtime that associates each of the bennchmark task ids to the average time it takes to evaluate
it 200 times. If for synthetic tasks the runtime is below 1 second, it can reach 3h30 for the most time-consuming
real-world taskk (MIG Flow synthesis), which should be taken into account when allocating the resources (e.g.
parallelizing on the seeds of the most time-consuming black-box). Note that the evaluation of each of the black-box can
be done on a single core.

| Task ID   (Comb (C)) / Mixed (M)) | Avg. time for 1 eval. (mm:ss) | Avg. time for 200 eval. (hh:mm:ss) | 
|-----------------------------------|-------------------------------|------------------------------------| 
| ackley        (C)                 | 00:00                         | 00:00:00                           | 
| ackley-53       (M)               | 00:00                         | 00:00:00                           | 
| rna_inverse_fold  (C)             | 00:00                         | 00:00:00                           | 
| pest                (C)           | 00:00                         | 00:00:01                           | 
| aig_optimization     (C)          | 00:04                         | 00:15:32                           | 
| xgboost_opt          (M)          | 00:05                         | 00:17:42                           | 
| antibody_design      (C)          | 00:13                         | 00:45:00                           | 
| svm_opt              (M)          | 00:20                         | 01:06:55                           | 
| aig_optimization_hyp (M)          | 00:32                         | 01:48:15                           | 
| mig_optimization     (C)          | 01:05                         | 03:37:36                           | 

---

## Runtime estimation

To estimate the optimization runtime for a fixed optimizer, task, and evaluation budget, call
function `estimate_runtime` from [runtime_estimator.py](../analysis/runtime_estimator.py) as shown in the example below.

```python
import torch

from analysis.runtime_estimator import estimate_runtime
from mcbo.optimizers.bo_builder import BO_ALGOS
from mcbo.task_factory import task_factory

algo_name = "Casmopolitan"
total_budget = 200
interpolation_num_samples = 5
task = task_factory('ackley', num_dims=[10, 5, 5], variable_type=['nominal', 'int'], num_categories=[15, None])

bo_builder = BO_ALGOS[algo_name]
optimizer = bo_builder.build_bo(search_space=task.get_search_space(), n_init=20, device=torch.device("cuda"))

estimate_runtime(optimizer=optimizer, task=task, total_budget=total_budget,
                 interpolation_num_samples=interpolation_num_samples)
```

Running it for standard optimizers we compared the estimated runtime with the actual runtime (
check [test_runtime_estimator.py](../tests/analysis/test_runtime_estimator.py)).

| Algo         | Estimated runtime (mm:ss) | Actual runtime (mm:ss) | Runtime to do the estimation (mm:ss) |
|--------------|---------------------------|------------------------|--------------------------------------|
| BOCS         | 27:03                     | 25:01                  | 00:45                                |
| BOiLS        | 12:38                     | 13:14                  | 00:21                                |
| BOSS         | 07:58                     | 08:48                  | 00:13                                |
| Casmopolitan | 06:43                     | 07:36                  | 00:11                                |
| COMBO        | 48:29                     | 39:05                  | 01:36                                |
| BODi         | 08:34                     | 08:39                  | 00:14                                |
| CoCaBO       | 15:11                     | 15:27                  | 00:25                                |

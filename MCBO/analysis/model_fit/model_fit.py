import os

import numpy as np
import pandas as pd
import torch

from mcbo.optimizers import BoBuilder
from mcbo.utils.general_utils import load_w_pickle, get_project_root
from mcbo.utils.model_utils import move_model_to_device
from mcbo.utils.experiment_utils import get_task_from_id


def get_fit_result_dir(task_id: str, data_method_source: str, surrogate_id: str, seed: int, n_observe: int):
    return f"{get_project_root()}/fit_results/{task_id}/{data_method_source}/{n_observe}-{seed}/{surrogate_id}/"


def get_model_fit(task_id: str, data_method_source: str, surrogate_id: str, seed: int, n_observe: int, device: int,
                  absolut_dir: str = None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dtype = torch.float64
    if data_method_source == "ga":
        data_method_source = "Genetic Algorithm"
    elif data_method_source == "sa":
        data_method_source = "Simulated Annealing"
    elif data_method_source == "rs":
        data_method_source = "Random Search"
    else:
        raise ValueError(data_method_source)
    result_dir = get_fit_result_dir(task_id=task_id, data_method_source=data_method_source, surrogate_id=surrogate_id,
                                    seed=seed, n_observe=n_observe)
    task = get_task_from_id(task_id=task_id, absolut_dir=absolut_dir)
    search_space = task.get_search_space(dtype=dtype)

    all_ys = pd.read_csv(f"{get_project_root()}/results/{task.name}/{data_method_source}/seed_{seed}_results.csv")[
        "f(x)"].values.reshape(-1, 1)
    all_xs = pd.DataFrame.from_dict(
        load_w_pickle(f"{get_project_root()}/results/{task.name}/{data_method_source}/seed_{seed}_x.pkl"))

    n_obs = n_observe
    train_inds = np.arange(n_obs)
    test_inds = np.arange(n_obs, len(all_xs))
    assert len(train_inds) + len(test_inds) == 200, len(train_inds) + len(test_inds)

    x_train = all_xs.iloc[train_inds]
    x_test = all_xs.iloc[test_inds]

    y_train = all_ys[train_inds]
    y_test = all_ys[test_inds]

    cases = {"train": (x_train, y_train), "test": (x_test, y_test)}

    input_constraints = None

    if device < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{device}")

    tr_id = None
    opt_builder = BoBuilder(model_id=surrogate_id, acq_opt_id="sa", acq_func_id="ei",
                            tr_id=tr_id)

    opt_instance = opt_builder.build_bo(
        search_space=search_space,
        n_init=1,
        input_constraints=input_constraints,
        dtype=dtype,
        device=device,
    )

    opt_instance.observe(x=x_train, y=y_train)

    data_buffer = opt_instance.data_buffer

    move_model_to_device(model=opt_instance.model, data_buffer=data_buffer, target_device=opt_instance.device)

    # Used to conduct and pre-fitting operations, such as creating a new model
    opt_instance.model.pre_fit_method(x=data_buffer.x, y=data_buffer.y)

    # Fit the model
    opt_instance.model.fit(x=data_buffer.x, y=data_buffer.y)

    os.makedirs(result_dir, exist_ok=True)

    for case in cases:
        fit_path = os.path.join(result_dir, f"{case}_fit.csv")
        x_path = os.path.join(result_dir, f"{case}_x.csv")
        if os.path.exists(fit_path) and os.path.exists(x_path):
            print(f"Already exists: {fit_path}")
            continue
        x, y = cases[case]
        opt_instance.model.to(device=torch.device("cpu"))
        opt_mu, opt_var = opt_instance.model.predict(search_space.transform(x).to(device="cpu"))
        # opt_mu, opt_var = opt_instance.model.predict(search_space.transform(x).to(device=opt_instance.device))
        log_likelihoods = - torch.nn.functional.gaussian_nll_loss(input=opt_mu, target=torch.tensor(y).to(opt_mu),
                                                                  var=opt_var, full=True, reduction='none')
        opt_mu = opt_mu.cpu().detach().numpy().flatten()
        opt_std = opt_var.cpu().detach().sqrt().numpy().flatten()
        log_likelihoods = log_likelihoods.detach().cpu().numpy().flatten()

        # save
        df = pd.DataFrame(columns=["y", "mean", "std", "log-likelihood"])
        df["y"] = y.flatten()
        df["mean"] = opt_mu
        df["std"] = opt_std
        df["log-likelihood"] = log_likelihoods
        df.to_csv(fit_path)
        x.to_csv(x_path)
        print(f"Done: {fit_path}")

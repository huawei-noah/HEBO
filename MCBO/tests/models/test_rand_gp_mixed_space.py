import random
import torch
from gpytorch.lazy import delazify

from mcbo.task_factory import task_factory
from mcbo.models.gp.rand_decomposition_gp import RandDecompositionGP

if __name__ == '__main__':
    dtype = torch.float64
    device = torch.device('cpu')
    task = task_factory('svm_opt')
    search_space = task.get_search_space(dtype=dtype)

    x_train_pd = search_space.sample(5)
    x_train = search_space.transform(x_train_pd)
    y_train = torch.tensor(task(x_train_pd))

    x_test_pd = search_space.sample(1)
    x_test = search_space.transform(x_test_pd)
    y_test = torch.tensor(task(x_test_pd))

    model = RandDecompositionGP(search_space, 1, pred_likelihood=False,  random_tree_size=1, batched_kernel=True)
    model_copy = RandDecompositionGP(search_space, 1, pred_likelihood=False,  random_tree_size=1, batched_kernel=True)

    model.fit(x_train, y_train)
    model_copy.fit(x_train, y_train, fixed_graph=model.graph)

    total_mean = 0
    total_var = 0   
    total_kern = 0

    prev_cov_cache = None
    
    for clique in model.graph:

        mu, var = model.partial_predict(x_test, clique)
        
        total_mean += mu
        total_var += var

        if prev_cov_cache is not None:
            assert torch.all(model.gp.prediction_strategy.covar_cache == prev_cov_cache)
        
        prev_cov_cache = model.gp.prediction_strategy.covar_cache

        x_test_pert = x_test.clone()

        for dim in range(x_test_pert.shape[1]):
            if dim not in clique:
                x_test_pert[:, dim] += random.random()

        mu_pert, var_pert = model.partial_predict(x_test_pert, clique)

        assert mu == mu_pert
        assert var == var_pert

        kern = model.kernel(x_test, x_train, clique=clique)
        total_kern += delazify(kern)

    mu, var = model.predict(x_test)
    kern = delazify(model.kernel(x_test, x_train))

    assert abs(total_mean.item() - mu.item()) < 0.01 * abs(total_mean.item())
    assert total_var.item() >= var.item()

    mu_control, var_control = model_copy.predict(x_test)

    assert abs(mu.item() - mu_control.item()) < 0.01 * abs(mu_control.item())
    assert abs(var.item() - var_control.item()) < 0.01 * var_control.item()

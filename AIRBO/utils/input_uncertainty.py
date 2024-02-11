from model_utils.input_transform import additional_xc_samples

import numpy as np
import torch
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from typing import Tuple, Union
from abc import ABCMeta, abstractmethod


def angle_2_coordinates(angle, radius, center):
    """
    Given the radius and center, translate an angle to coordinates
    """
    x = radius * np.cos(angle) + center[0]
    y = radius * np.sin(angle) + center[1]
    return x, y


def to_cicular_distribution(angle_vals, radius, center):
    """
    Given a list of values in range of [0, 2*pi], return their coordinates
    """
    _angles = np.expand_dims(angle_vals, axis=-1)
    x = np.concatenate(
        angle_2_coordinates(_angles, radius=radius, center=center),
        axis=-1
    )
    return x


class InputDistribution(metaclass=ABCMeta):
    def __init__(self, name=None):
        self._est_mean = None
        self._est_var = None
        self.name = name if name is not None else self.__class__.__name__
        self.n_var = 1

    @property
    def estimated_mean(self):
        return self._est_mean

    @property
    def estimated_var(self):
        return self._est_var

    @abstractmethod
    def sample(self, size: Tuple, x=None, **kwargs):
        """
        sample from the distribution
        :param size: a tuple of B * h, where B is the batch size and h is the sampling size
        :param x: the corresponding x, used for location-sensitive distribution
        :param kwargs: sampling configurations
        :return:
        """
        raise NotImplementedError()


class ScipyInputDistribution(InputDistribution):
    def __init__(self, distrib, name, n_var=None):
        super(ScipyInputDistribution, self).__init__(name=name)
        self.model = distrib
        self.n_var = n_var
        if self.n_var is None:
            _s = np.atleast_1d(self.model.rvs())
            self.n_var = len(_s)

        # update the estimated distribution
        _samp_size = 10 ** 5
        self.samples = self.model.rvs(_samp_size)
        self._est_mean = self.samples.mean(axis=0)
        self._est_var = self.samples.var(axis=0)

    def sample(self, size, x=None, **kwargs):
        return self.model.rvs(size=size, **kwargs).reshape(
            (*size, self.n_var) if isinstance(size, Tuple) else (size, self.n_var)
        )


class Circular2Distribution(InputDistribution):
    """
    A 2D circular distribution
    """

    def __init__(self, distrib, name, n_var=2, radius=1.0, center=(0, 0)):
        super(Circular2Distribution, self).__init__(name=name)
        assert n_var == 2
        self.model = distrib
        self.n_var = self.model.dim if n_var is None else n_var
        self.radius = radius
        self.center = center

        # update the estimated distribution
        _samp_size = 10 ** 5
        _angles = self.model.rvs(_samp_size) * np.pi * 2.0
        self.samples = to_cicular_distribution(_angles, self.radius, self.center)
        self._est_mean = self.samples.mean(axis=0)
        self._est_var = self.samples.var(axis=0)

    def sample(self, size, x=None, **kwargs):
        _angles = self.model.rvs(size) * np.pi * 2.0
        x = to_cicular_distribution(_angles, self.radius, self.center)
        return x


class ConcatDistribution(InputDistribution):
    def __init__(self, distribs, name, n_var=None):
        super(ConcatDistribution, self).__init__(name=name)
        self.model = distribs

        # update the estimated distribution
        _samp_size = 10 ** 5
        self.samples = np.concatenate(
            [d.sample(_samp_size) for d in self.model],
            axis=-1
        )
        self._est_mean = self.samples.mean(axis=0)
        self._est_var = self.samples.var(axis=0)

    def sample(self, size, x=None, **kwargs):
        x = np.concatenate(
            [
                d.sample(size).reshape(
                    (*size, d.n_var) if isinstance(size, Tuple) else (size,  d.n_var)
                )
                for d in self.model
            ],
            axis=-1
        )
        return x


class StepInputDistribution(InputDistribution):
    def __init__(self, distribs, name, n_var=None):
        super(StepInputDistribution, self).__init__(name=name)
        self.models = distribs
        self.n_var = self.models[0].dim if n_var is None else n_var

        # update the estimated distribution
        _samp_size = 10 ** 5
        self.samples = np.concatenate([m.rvs(_samp_size) for m in self.models], axis=0)
        self._est_mean = self.samples.mean(axis=0)
        self._est_var = self.samples.var(axis=0)

    def sample(self, size, x=None, **kwargs):
        samp1 = self.models[0].rvs(size=size, **kwargs).reshape(*size, self.n_var)
        samp2 = self.models[1].rvs(size=size, **kwargs).reshape(*size, self.n_var)
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = x[..., None, :]  # broadcast to sample size: B, h, d
        ret = np.where(x < 0.6, samp1, samp2)
        return ret


class VaryingInputDistribution(InputDistribution):
    def __init__(self, rvs, name, n_var=None):
        super(VaryingInputDistribution, self).__init__(name=name)
        self.n_var = self.models[0].dim if n_var is None else n_var
        self.rvs_func = rvs

        # update the estimated distribution
        self.samples = None
        self._est_mean = None
        self._est_var = None

    def sample(self, size, x=None, **kwargs):
        if isinstance(size, Tuple):
            batch_size = size[:-1]
            sampling_size = size[-1]
        else:
            batch_size = []
            sampling_size = size
        total_batch_size = np.prod(batch_size) if len(batch_size) > 0 else 1
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        total_x = x.reshape(-1, x.shape[-1]).squeeze(-1)
        batch_samples = []
        for _i in range(total_batch_size):
            _x = total_x[_i].squeeze()
            a, b = 0.5, 0.5
            scale = (np.sin(np.pi * 4 * _x) + 1.0) * 0.09
            # scale = (2.0-_x*2.0+1e-4)*0.09
            batch_samples.append(self.rvs_func(a=a, b=b, scale=scale, size=sampling_size))
        samples = np.stack(batch_samples).reshape(*batch_size, sampling_size, -1)

        return samples


class GMMInputDistribution(InputDistribution):
    def __init__(self, n_components=2,
                 mean=np.array([[-0.1], [0.05]]),
                 covar=np.array([0.0007, 0.007]),
                 weight=np.array([0.4, 0.6]),
                 covariance_type='spherical'):
        super(GMMInputDistribution, self).__init__()
        self.model = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
        self.model.means_ = mean
        self.model.covariances_ = covar
        self.model.weights_ = weight

        # update the estimated distribution
        _samp_size = 10 ** 5
        self.samples, _ = self.model.sample(_samp_size)
        self.n_var = len(self.samples[0])
        self._est_mean = self.samples.mean(axis=0)
        self._est_var = self.samples.var(axis=0)

    def sample(self, size, x=None, **kwargs):
        batch_size = size[:-1]
        sampling_size = size[-1]
        total_batch_size = np.prod(batch_size) if len(batch_size) > 0 else 1
        batch_samples = []
        for _ in range(total_batch_size):
            batch_samples.append(self.model.sample(sampling_size)[0])
        samples = np.stack(batch_samples).reshape(*batch_size, sampling_size, -1)
        return samples


# %%
if __name__ == "__main__":
    # %%
    import numpy as np
    from problems.problem_factory import get_test_problem

    cmap = plt.get_cmap("coolwarm")
    min_cov = 1e-6
    n_var = 1
    distribution_candidates = [
        # GaussianInput(n_var=1, mean=0.0, cov=0.01 ** 2.0),
        # ScipyDistributionInput(stats.chi2(loc=0.0, scale=0.01, df=2)),
        # GMMInput(n_components=2, mean=np.array([[-0.017], [0.03]]),
        #          covar=np.array([0.010 ** 2.0, 0.017 ** 2.0]), weight=np.array([0.5, 0.5])),
        # ScipyDistributionInput(stats.truncnorm(loc=0.0, scale=0.02, a=0.0, b=2)),
        # ScipyInputDistribution(stats.beta(a=0.4, b=0.4, scale=1.0, loc=-0.5), name='beta', n_var=1),
        # Circular2Distribution(stats.uniform(), name='circular2D', n_var=2),
        ScipyInputDistribution(stats.beta(a=0.4, b=0.4, loc=0.1*-0.5, scale=0.1), name='beta', n_var=1),
        # ScipyInputDistribution(stats.uniform(loc=-0.15, scale=0.3), name='uniform', n_var=1),
        # GMMInputDistribution(
        #     n_components=2, mean=np.array([[0, 0,], [-1, 1]]),
        #     covar=np.array([
        #         [0.1 ** 2, -0.3 ** 2 ],
        #         [-0.3 ** 2, 0.1 ** 2],
        #     ]),
        #     covariance_type='tied',
        #     weight=np.array([0.5, 0.5])
        # )

    ]

    nrows, ncols, ax_scale = 1, len(distribution_candidates), 1.6
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                             figsize=(4 * ncols * ax_scale, 3 * nrows * ax_scale))
    axes = axes.flatten()
    for d_i, distrib in enumerate(distribution_candidates):
        x_samples = getattr(distrib, "samples", None)
        if x_samples is None:
            x_samples = distrib.sample(10 ** 5)

        est_norm = stats.norm(
            loc=distrib.estimated_mean, scale=distrib.estimated_var ** 0.5
        ) if distrib.n_var == 1 else stats.multivariate_normal(
            mean=distrib.estimated_mean, cov=np.diag(distrib.estimated_var)
        )

        if distrib.n_var == 1:
            mesh_coords = np.linspace(x_samples.min(), x_samples.max(), 1000)
            axes[d_i].hist(x_samples, bins=100, edgecolor='k')
            twin_ax = plt.twinx(axes[d_i])
            twin_ax.plot(mesh_coords, est_norm.pdf(mesh_coords), color='r',
                         label=f'est_norm: \n'
                               f'$\mu={est_norm.mean()}$, \n'
                               f'$\sigma={est_norm.std()}$')
            twin_ax.legend(loc='upper right')
        elif distrib.n_var == 2:
            x_range = (
                min(x_samples[:, 0].min(), x_samples[:, 1].min()),
                max(x_samples[:, 0].max(), x_samples[:, 1].max()),
            )
            mesh_x = np.linspace(x_range[0], x_range[1], 1000)
            mesh_y = np.linspace(x_range[0], x_range[1], 1000)
            xx, yy = np.meshgrid(mesh_x, mesh_y)
            pos = np.dstack((xx, yy))
            pc = axes[d_i].contour(mesh_x, mesh_y, est_norm.pdf(pos), cmap=cmap)
            axes[d_i].clabel(pc)

            axes[d_i].scatter([s[0] for s in x_samples],
                              [s[1] for s in x_samples], label='samples',
                              marker='.', alpha=0.5, color='tab:grey')

            axes[d_i].set_xlim(x_range)
            axes[d_i].set_ylim(x_range)
        else:
            print("can not visualize ")
    fig.tight_layout()
    plt.show()

    # %%
    # define problem
    raw_func_name = 'RKHS-S'
    prob = get_test_problem(
        raw_func_name, n_var=n_var, crpt_method="raw", mesh_sample_num=1000,
        input_range_offset=0, x_offset=0
    )

    # %%
    # find exact optimum
    optimum_results = {}
    exact_opt_ind = np.argmin(prob.mesh_vals) if prob.minimization else np.argmax(prob.mesh_vals)
    exact_opt_x = prob.mesh_coords[exact_opt_ind:exact_opt_ind + 1]
    exact_opt_y = prob.mesh_vals[exact_opt_ind:exact_opt_ind + 1]
    optimum_results['exact'] = (exact_opt_x, exact_opt_y, exact_opt_y, prob.mesh_vals)
    print(f"[exact]opt_x={exact_opt_x}, opt_y={exact_opt_y}")

    # find the expected optimum
    num_expectation_eval = 100
    for distrib in distribution_candidates:
        est_norm = ScipyInputDistribution(
            stats.norm(loc=distrib.estimated_mean, scale=distrib.estimated_var ** 0.5)
            if distrib.n_var == 1
            else stats.multivariate_normal(mean=distrib.estimated_mean, cov=distrib.estimated_var),
            name="est_norm", n_var=n_var
        )
        for _d_name, _sampling_func in [('true', distrib.sample),
                                        ('est_norm', est_norm.sample)]:
            print(f"Computing {distrib.name}-{_d_name} optimum...")
            mesh_expectations = prob.evaluate(
                additional_xc_samples(
                    prob.mesh_coords, num_expectation_eval, n_var, _sampling_func
                ).reshape(-1, n_var)
            ).reshape(prob.mesh_coords.shape[0], -1).mean(axis=-1)

            _opt_ind = np.argmin(mesh_expectations) if prob.minimization \
                else np.argmax(mesh_expectations)
            _opt_expected_y = mesh_expectations[_opt_ind]
            _opt_x = prob.mesh_coords[_opt_ind: _opt_ind + 1]
            _opt_y = prob.evaluate(_opt_x)
            optimum_results[_d_name] = (_opt_x, _opt_y, _opt_expected_y, mesh_expectations)
            print(f"[{_d_name}] opt_x={_opt_x}, opt_expected_y={_opt_expected_y}")

        # vis
        ax_scale = 2.5
        fig, ax = plt.subplots(squeeze=True, figsize=(4 * ax_scale, 3 * ax_scale))
        ax.plot(prob.mesh_coords, prob.mesh_vals, c='grey', label='gdt', alpha=0.5)
        markers = ['*', 'o', 'd']
        colors = ['k', 'm', 'c', 'r', 'b', 'g']
        for opt_i, (opt_name, (opt_x, opt_y, opt_expected_y, prob.mesh_vals)) in enumerate(
                optimum_results.items()):
            ax.plot(prob.mesh_coords, prob.mesh_vals, c=colors[opt_i], alpha=0.5)
            ax.scatter(opt_x.flatten(), opt_y.flatten(), label=f"{opt_name}",
                       marker=markers[opt_i], c=colors[opt_i], s=30)
            ax.axvline(opt_x.squeeze(), c=colors[opt_i], alpha=0.5, ls='--')
        ax.legend()
        # fig.suptitle(f"double-peak@{distrib.name}(0.4, 0.2, 0.1)")
        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()




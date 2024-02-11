"""
Environment of robot pushing
"""
from problems.robot_pushing import push_world as pw

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import Tuple, List
from abc import abstractmethod, ABC


def step_func(x):
    y = np.where(x <= 1., 0, x * 0.7)
    return y


def squared_func(x, a=1.0):
    return (x * a) ** 2.0


def linear_func(x):
    return x


def narrow_exp_func(x):
    y = -np.exp(-(x * 2.9) ** 2.0) + 1.0 + x * 1
    return y


def neg_exp_func(x):
    return -np.exp(-(x * 2.9) ** 2.0) + 1.0 + x * 0.1


class BasicPushEnv(ABC):
    def __init__(self, goals: List[np.array] = None, show_gui: bool = False):
        """
        Basic push environment
        :param goals: the target positions
        :param show_gui: whether to show the GUI
        """
        self.show_gui = show_gui
        self.obj_shape = 'circle'
        self.obj_size = 1
        self.obj_friction = 0.01
        self.obj_density = 0.05
        self.obj_pos = (0, 0)
        self.bfriction = 0.01

        self.robot_shape = 'rectangle'
        self.robot_size = (0.3, 1.0)

        self.goals = goals
        self.n_vars = None

    def get_goals(self):
        return self.goals

    @abstractmethod
    def push(self, *args):
        pass

    def compute_dist(self, goal_pos, end_pos):
        return np.linalg.norm(goal_pos, end_pos)

    def evaluate_ex(self, X):
        end_pos = self.push(*X)
        goal_pos = self.get_goals()
        _dist_val = self.compute_dist(goal_pos, end_pos)
        dist = np.array([[_dist_val]])

        return goal_pos, end_pos, dist

    def evaluate(self, X):
        goal_pos, end_pos, dist = self.evaluate_ex(X)
        return dist


class Push3Env(BasicPushEnv):
    def __init__(self, goals=None, show_gui: bool = False):
        """
        the push environment with 3 parameters: rx, ry, rt
        """
        super(Push3Env, self).__init__(goals, show_gui)
        self.n_vars = 3

    def push(self, rx: float, ry: float, rt: int):
        """
        let the robot push the box, the push angle is determined automatically
        :param rx: the x of robot initial position
        :param ry: the y of robot initial position
        :param rt: the pushing time
        :return: the ending position of the box
        """
        world = pw.b2WorldInterface(self.show_gui)
        thing, base = pw.make_thing(
            500, 500, world, self.obj_shape, self.obj_size, self.obj_friction,
            self.obj_density, self.bfriction, self.obj_pos
        )

        push_angle = np.arctan(ry / rx)
        robot = pw.end_effector(world, (rx, ry), base, push_angle,
                                self.robot_shape, self.robot_size)
        real_rt = int(rt * 10)
        end_pos = pw.simu_push(world, thing, robot, base, real_rt)

        return end_pos


class DualGoalsP3Env(Push3Env):
    def __init__(self, goals=None, show_gui: bool = False):
        if goals is None:
            goals = [np.array([-3, 3]).reshape(1, -1),
                     np.array([3, -3]).reshape(1, -1)]
        super(DualGoalsP3Env, self).__init__(goals, show_gui)

    def compute_dist(self, goal_pos, end_pos):
        return min(narrow_exp_func(np.linalg.norm(goal_pos[0] - end_pos)),
                   np.linalg.norm(goal_pos[1] - end_pos))


class TripleGoalsP3Env(Push3Env):
    def __init__(self, goals: List[np.array] = None, show_gui: bool = False):
        if goals is None:
            goals = [np.array([-3, 3]).reshape(1, -1),
                     np.array([3, -3]).reshape(1, -1),
                     np.array([4.3, 4.3]).reshape(1, -1),
                     np.array([5.1, 3.0]).reshape(1, -1)]
        super(TripleGoalsP3Env, self).__init__(goals, show_gui)

    def compute_dist(self, goal_pos, end_pos):
        dist = min(
            squared_func(np.linalg.norm(goal_pos[0] - end_pos)),
            min(
                linear_func(np.linalg.norm(goal_pos[1] - end_pos)),
                min(
                    linear_func(np.linalg.norm(goal_pos[2] - end_pos)),
                    linear_func(np.linalg.norm(goal_pos[3] - end_pos)),
                )
            )
        )
        return dist


class TripleGoalsP3Env_v2(Push3Env):
    def __init__(self, goals: List[np.array] = None, show_gui: bool = False):
        if goals is None:
            goals = [np.array([-3, 3]).reshape(1, -1),
                     np.array([3, -3]).reshape(1, -1),
                     np.array([2.3, 4.3]).reshape(1, -1),
                     np.array([3.1, 3.0]).reshape(1, -1)]
        super(TripleGoalsP3Env_v2, self).__init__(goals, show_gui)

    def compute_dist(self, goal_pos, end_pos):
        dist = min(
            squared_func(np.linalg.norm(goal_pos[0] - end_pos), 2.0),
            min(
                min(
                    squared_func(np.linalg.norm(goal_pos[2] - end_pos), 3.0),
                    squared_func(np.linalg.norm(goal_pos[3] - end_pos), 3.0)
                ),
                squared_func(np.linalg.norm(goal_pos[1] - end_pos), 1.0),
            )
        )
        return dist


class Push4Env(BasicPushEnv):
    def __init__(self, goals: List[np.array] = None, show_gui: bool = False):
        """
        Push environment with 4 parameters: rx, ry, rt, ra
        """
        super(Push4Env, self).__init__(goals, show_gui)
        self.n_vars = 4

    def push(self, rx: float, ry: float, rt: int, ra: float):
        """
        let the robot push the box
        :param rx: x of the robot initial position
        :param ry: y of the robot initial position
        :param rt: pushing time, 0~30
        :param ra: pushing angle,-np.pi/2 ~ np.pi/2
        :return: the ending position of box after push
        """
        world = pw.b2WorldInterface(self.show_gui)
        thing, base = pw.make_thing(
            500, 500, world, self.obj_shape, self.obj_size, self.obj_friction,
            self.obj_density, self.bfriction, self.obj_pos
        )

        push_angle = ra
        robot = pw.end_effector(world, (rx, ry), base, push_angle,
                                self.robot_shape, self.robot_size)
        real_rt = int(rt * 10)
        end_pos = pw.simu_push(world, thing, robot, base, real_rt)

        return end_pos


class TripleGoalsP4Env(Push4Env):
    def __init__(self, goals: List[np.array] = None, show_gui: bool = False):
        if goals is None:
            goals = [np.array([-3, 3]).reshape(1, -1),
                     np.array([3, -3]).reshape(1, -1),
                     np.array([4.3, 4.3]).reshape(1, -1),
                     np.array([5.1, 3.0]).reshape(1, -1)]
        super(TripleGoalsP4Env, self).__init__(goals, show_gui)

    def compute_dist(self, goal_pos, end_pos):
        dist = min(
            squared_func(np.linalg.norm(goal_pos[0] - end_pos)),
            min(
                linear_func(np.linalg.norm(goal_pos[1] - end_pos)),
                min(
                    linear_func(np.linalg.norm(goal_pos[2] - end_pos)),
                    linear_func(np.linalg.norm(goal_pos[3] - end_pos)),
                )
            )
        )
        return dist


class DualGoalsP4Env(Push4Env):
    def __init__(self, goals=None, show_gui: bool = False):
        if goals is None:
            goals = [np.array([-3, 3]).reshape(1, -1),
                     np.array([3, -3]).reshape(1, -1)]
        super(DualGoalsP4Env, self).__init__(goals, show_gui)

    def compute_dist(self, goal_pos, end_pos):
        return min(narrow_exp_func(np.linalg.norm(goal_pos[0] - end_pos)),
                   np.linalg.norm(goal_pos[1] - end_pos))


def plot_push_world(goals, obj_end_pos, dist, rx, ry, rt, obj_init_pos=(0.0, 0.0),
                    x_range=None, y_range=None,
                    plot_goals=True,
                    plot_obj_init_pos=True, plot_obj_end_pos=True,
                    plot_robot_init_pos=True, plot_push_direction=True,
                    fig=None, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # goals
    if not isinstance(goals, List):
        goals = [goals]

    if plot_goals:
        for g in goals:
            ax.scatter([g.flatten()[0]], [g.flatten()[1]], label="goal",
                       marker='*', c='y')

    # robot push
    if plot_push_direction:
        ax.arrow(rx, ry, obj_end_pos[0] - rx, obj_end_pos[1] - ry, ls='--', color='tab:grey',
                 alpha=0.5, length_includes_head=True, head_width=0.2, lw=0.5)

    # obj end pos
    if plot_obj_end_pos:
        ax.scatter([obj_end_pos[0]], [obj_end_pos[1]], label='obj_end_pos',
                   marker='o', facecolor='none', edgecolor='b', s=15)
    # obj init pos
    if plot_obj_init_pos:
        ax.scatter([obj_init_pos[0]], [obj_init_pos[1]], label='obj_init_pos',
                   marker='o', facecolor='none', edgecolor='tab:grey', s=15)

    # robot pos
    if plot_robot_init_pos:
        ax.scatter([rx], [ry], label='robot_pos', marker='s', facecolor='none', edgecolor='k',
                   s=15)



    if x_range is not None:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(x_range)

    return fig, ax


def plot_env_landscape(env: BasicPushEnv, x_range: Tuple, y_range: Tuple, plot_3d=False,
                       fig=None, ax=None, **plot_kwargs):
    """
    plot the landscape of loss values
    """
    mesh_X = np.arange(x_range[0], x_range[1], 0.1)
    mesh_Y = np.arange(y_range[0], y_range[1], 0.1)
    mesh_xx, mesh_yy = np.meshgrid(mesh_X, mesh_Y)
    mesh_coords = np.stack([mesh_xx, mesh_yy], axis=-1)
    mesh_vals = np.array(
        [env.compute_dist(env.get_goals(), e) for row in mesh_coords for e in row]
    ).reshape(len(mesh_X), len(mesh_Y))

    if ax is None:
        ax_scale = 1.5
        fig = plt.figure(figsize=(4 * ax_scale, 3 * ax_scale))
        ax = fig.add_subplot(111, projection="3d" if plot_3d else None)

    if plot_3d:
        pc = ax.plot_surface(mesh_xx, mesh_yy, mesh_vals, cmap=cmap)
    else:
        levels = plot_kwargs.get("levels", None)
        pc = ax.contourf(mesh_X, mesh_Y, mesh_vals, levels=levels, cmap=cmap)
    plt.colorbar(pc, ax=ax)

    fig.tight_layout()
    return fig, ax


# %%
if __name__ == "__main__":
    # %%
    import matplotlib.pyplot as plt
    from tqdm.auto import trange
    import math
    from functools import partial

    cmap = plt.get_cmap("coolwarm_r")

    # %%
    # show the PushWorld with GUI
    # rx, ry, rt = 3, -3, 24
    # env = TripleGoalsP4Env(show_gui=True)
    # env.push(rx, ry, rt, 0)

    # %%
    # plot several functions
    xx = np.arange(-6, 6, 0.1)
    plt.figure()
    for fn, func in [
        ('linear', linear_func),
        ('square_a0.5', partial(squared_func, a=0.5)),
        ('square_a1', partial(squared_func, a=1)),
        ('square_a2', partial(squared_func, a=2)),
        ('square_a3', partial(squared_func, a=3)),
    ]:
        plt.plot(xx, func(abs(xx)), label=f'{fn}')
    plt.ylim(0., 10)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # %%
    # plot the env landscape
    env = TripleGoalsP3Env_v2()
    x_range = (-6, 6)
    y_range = (-6, 6)
    fig = plt.figure(figsize=(4 * 1.5 * 2, 3 * 1.5))
    ax0 = fig.add_subplot(121)
    plot_env_landscape(env, x_range, y_range, fig=fig, ax=ax0)
    ax1 = fig.add_subplot(122, projection="3d")
    plot_env_landscape(env, x_range, y_range, plot_3d=True, fig=fig, ax=ax1)
    plt.show()

    # %%
    # try the PushWorld, compute the statistics of the results
    env = TripleGoalsP3Env_v2()
    samp_num = 10
    rxs = np.random.uniform(-5, 5, samp_num)
    rys = np.random.uniform(-5, 5, samp_num)
    rts = np.random.uniform(1, 30, samp_num)
    results = [env.evaluate_ex((rxs[i], rys[i], rts[i])) for i in trange(samp_num)]

    ncols = 4
    nrows = int(math.ceil(samp_num / ncols))
    ax_scale = 1.
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, squeeze=True,
        figsize=(4 * ncols * ax_scale, 3 * nrows * ax_scale)
    )
    axes = axes.flatten()
    for samp_i, (_rx, _ry, _rt, _r) in enumerate(zip(rxs, rys, rts, results)):
        goals, end_pos, dist = _r
        plot_push_world(goals, end_pos, dist, _rx, _ry, _rt, fig=fig, ax=axes[samp_i])
    plt.show()

    # %%
    # try to push under input uncertainty
    from utils import input_uncertainty as iu
    from tqdm.auto import tqdm
    import seaborn as sns

    min_cov = 1e-6
    env = TripleGoalsP3Env_v2()
    input_distrib = iu.GMMInputDistribution(
        n_components=2, mean=np.array([[0, 0, 0], [-1, 1, 0]]),
        covar=np.array([
            [0.1 ** 2, -0.3 ** 2, min_cov, ],
            [-0.3 ** 2, 0.1 ** 2, min_cov, ],
            [min_cov, min_cov, 1 ** 2.0],
        ]),
        covariance_type='tied',
        weight=np.array([0.5, 0.5])
    )
    # input_distrib = iu.ScipyDistributionInput(
    #     stats.multivariate_normal(mean=[0., 0., 0.], cov=np.diag([1**2, 1**2, 1**2])),
    #     name="multivariate-normal"
    # )

    r_config_noises = input_distrib.sample(size=1000)
    push_results = {}
    for i, rt_mean in enumerate([10, 25, 30]):
        for (rx_mean, ry_mean) in [(-4, -4), (-4, 4)]:
            push_result = []
            push_results[(rx_mean, ry_mean, rt_mean)] = push_result
            for x_n in tqdm(r_config_noises):
                _rx = rx_mean + x_n[0]
                _ry = ry_mean + x_n[1]
                _rt = rt_mean + x_n[2]
                if env.n_vars == 4:
                    _ra = np.arctan(_ry / _rx) + x_n[3]
                    x = np.array([_rx, _ry, _rt, _ra])
                else:
                    x = np.array([_rx, _ry, _rt])
                ra_mean = np.arctan(ry_mean / rx_mean)
                goals, end_pos, dist = env.evaluate_ex(x)
                push_result.append((x, end_pos, dist))

    #  plot
    nrows, ncols, ax_scale = 2, 1, 1.8
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=True,
                             figsize=(4 * ncols * ax_scale, 3 * nrows * ax_scale))
    example_r = push_results[[k for k in push_results.keys()][0]]
    is_landscape_plotted = False
    for i, ((rx_mean, ry_mean, rt_val), _push_ret) in enumerate(push_results.items()):
        if rt_val == 10:
            if not is_landscape_plotted:
                plot_env_landscape(env, x_range, y_range, fig=fig, ax=axes[0],
                               levels=[0, 1, 2, 3, 4, 5, 10])
                is_landscape_plotted = True

            axes[0].scatter(
                [_x[0] for (_x, _ep, _dist) in _push_ret],
                [_x[1] for (_x, _ep, _dist) in _push_ret],
                label=f'r_pos', marker='.', alpha=0.5, s=5, c='tab:grey',
            )
            sns.kdeplot(
                [_x[0] for (_x, _ep, _dist) in _push_ret],
                [_x[1] for (_x, _ep, _dist) in _push_ret],
                ax=axes[0], shade=True, alpha=0.5, color='tab:grey',
            )

        axes[0].scatter(
            [_ep[0] for (_x, _ep, _dist) in _push_ret],
            [_ep[1] for (_x, _ep, _dist) in _push_ret],
            label=f'from ({rx_mean}, {ry_mean})-{rt_val}', marker='x', alpha=0.5, s=5
        )
        sns.kdeplot(
            [_ep[0] for (_x, _ep, _dist) in _push_ret],
            [_ep[1] for (_x, _ep, _dist) in _push_ret],
            ax=axes[0], shade=True, alpha=0.5
        )

        # plot rt
        axes[1].hist([_x[-1] for (_x, _ep, _dist) in _push_ret], bins=20, label=f'rt{rt_val}',
                     edgecolor='k')
    axes[0].set_xlim(-7, 7)
    axes[0].set_ylim(-7, 7)
    axes[0].set_title(f"{input_distrib.name}")
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.show()

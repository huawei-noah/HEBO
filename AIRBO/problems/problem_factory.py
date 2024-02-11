from problems.test_function import TestFunctions
from problems.rkhs import rkhs_synth

import matplotlib as mpl

# mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import math
import numpy as np
from pymoo.factory import get_problem
from scipy import signal
from scipy import stats
from functools import partial

BENCHMARK_FUNCTION_NAMES = [
    # 'Sin',
    # 'Ackley',
    # 'Griewank',
    # 'Rastrigin',
    # 'Schwefel',
    # 'Sphere',
    # 'Zakharov',
    # 'Sinc',
    # 'Sin',
    # 'CustomB',
    # 'CustomC',
    # 'CustomD',
    # 'CustomE',
    # 'CustomF',
    # 'CustomG',
    'CustomH',
    # 'CustomI',
]


def gauss_pulse_noise(x, fc, bw, offset, cutoff_thr=0.0):
    _s, _env = signal.gausspulse((x - offset), fc=fc, bw=bw, retenv=True)
    ns = _s.sum(axis=1, keepdims=True)
    nenv = (_env.sum(axis=1, keepdims=True) / x.shape[1])
    s = ns * (nenv >= cutoff_thr)
    return s


def step_func(x, xl, xu, loc=0.5):
    cutoff_val = (xu - xl) * loc + xl
    return np.where(x < cutoff_val, 0., 1.0)


def func_type_C(x, n_var, with_noise=True):
    raw_val = get_problem("Ackley", n_var=n_var).evaluate(x.reshape(-1, n_var))
    noise_1 = gauss_pulse_noise(x, fc=1, bw=0.1, offset=-28) * 12.75
    noise_2 = gauss_pulse_noise(x, fc=0.4, bw=0.15, offset=-12) * 3.45
    noise_3 = gauss_pulse_noise(x, fc=1.8, bw=0.05, offset=+15) * 5.42

    ret = (raw_val + noise_1 + noise_2 + noise_3) if with_noise else raw_val
    return ret


def func_type_D(x, n_var, level):
    m0 = get_problem("Ackley", n_var=n_var, a=20, b=1 / 5, c=2 * np.pi).evaluate(
        (x - 15).reshape(-1, n_var))
    n1 = gauss_pulse_noise(x, fc=1, bw=0.1, offset=-30) * 18.75
    n2 = gauss_pulse_noise(x, fc=0.4, bw=0.15, offset=-12) * 7.45
    n3 = gauss_pulse_noise(x, fc=1.8, bw=0.05, offset=+15) * 10.42
    m1 = -np.sinc(2 * np.pi * (x + 7) / 80.0) + 2.0
    if level == 0:
        ret = m0
    elif level == 1:
        ret = m0 + n1
    elif level == 2:
        ret = m0 + n1 + n2
    elif level == 3:
        ret = m0 + n1 + n2 + n3
    elif level == 4:
        ret = m0 * m1 + n1 + n2 + n3
    else:
        raise ValueError("Unsupported level:", level)
    return ret - 5.0


def func_type_E(x, xl, xu, n_var):
    m0 = -step_func(x, loc=0.5, xl=xl, xu=xu) + 1.0
    m1 = (m0 * step_func(x, loc=0.5, xl=xl, xu=xu)) * 0.2
    # m3 = -(m0*step_func(x, loc=0.5, xl=xl, xu=xu))*2+80.0
    n1 = np.sin(x * 2 - np.pi) * np.exp(-x / 20)
    # n2 = np.sin(x*2-np.pi)*5
    # n3 = guass_pulse_noise(x, fc=0.3, bw=0.09, offset=+35) * 0.1
    return -(np.sin(0.7 * x) / x) * 10 + m0 * n1


def func_type_E1(x, xl, xu, n_var):
    return (np.sin(x / 20) * x ** 2. / 177) * np.sin(x)


def func_type_G(x, xl, xu, n_var):
    m0 = -step_func(x, loc=0.7, xl=xl, xu=xu) + 1.0
    # m1 = -(m0*step_func(x, loc=0.5, xl=xl, xu=xu))*0.2+40.0
    # m3 = -(m0*step_func(x, loc=0.5, xl=xl, xu=xu))*2+80.0
    n1 = 50 * np.sin(x * 2 - np.pi) * np.exp(-x / 50)
    n2 = np.sin(x * 2 - np.pi) * 5
    # n3 = guass_pulse_noise(x, fc=0.3, bw=0.09, offset=+35) * 0.1
    return m0 * n1 + m0 * 200 + n2


def func_type_F(x, xl, xu, n_var):
    m0 = get_problem("Ackley", n_var=n_var, a=20, b=1 / 5, c=2 * np.pi).evaluate(
        (x - 15).reshape(-1, n_var))
    n1 = gauss_pulse_noise(x, fc=0.4, bw=0.1, offset=-30) * 18.75
    n2 = gauss_pulse_noise(x, fc=0.7, bw=0.1, offset=-8) * 7.45
    n3 = gauss_pulse_noise(x, fc=0.3, bw=0.09, offset=+35) * 20.42
    m1 = -np.sinc(2 * np.pi * (x + 7) / 80.0) + 2.0
    m2 = -step_func(x, loc=0.3, xl=xl, xu=xu) + 0.3
    m3 = -((-step_func(x, loc=0.9, xl=xl, xu=xu) + 1.0) * step_func(x, loc=0.7, xl=xl, xu=xu)) + 0.5
    m4 = (step_func(x, loc=0.9, xl=xl, xu=xu) + 10.2)

    # return m3
    return (m0 + n1 + n2 + n3) * m2 * m3


def func_type_H(x, n_var, with_noise=True):
    raw_val = get_problem("Ackley", n_var=n_var).evaluate(x.reshape(-1, n_var))
    noise_1 = gauss_pulse_noise(x, fc=0.5, bw=0.2, offset=-20) * 30.32
    noise_2 = gauss_pulse_noise(x, fc=0.4, bw=0.15, offset=-12) * 3.45
    noise_3 = gauss_pulse_noise(x, fc=1.8, bw=0.05, offset=+15) * 5.42
    noise_4 = gauss_pulse_noise(x, fc=0.5, bw=0.1, offset=+35) * 12.42

    ret = (raw_val + noise_1 + noise_2 + noise_3 + noise_4) if with_noise else raw_val
    return ret


def func_type_H1(x, n_var, with_noise=True):
    raw_val = get_problem("Ackley", n_var=n_var, a=20, b=1 / 5, c=0.1 * np.pi).evaluate(
        x.reshape(-1, n_var))
    noise_1 = gauss_pulse_noise(x, fc=0.38, bw=0.18, offset=-45) * 20.32
    noise_2 = gauss_pulse_noise(x, fc=0.23, bw=0.15, offset=-18) * 3.45
    noise_3 = gauss_pulse_noise(x, fc=0.31, bw=0.09, offset=+25) * 9.42
    noise_4 = gauss_pulse_noise(x, fc=0.27, bw=0.3, offset=+65) * 17.42

    ret = (raw_val + noise_1 + noise_2 + noise_3 + noise_4) if with_noise else raw_val
    return ret


def func_type_I(x, n_var, with_noise=True):
    raw_val = get_problem("Ackley", a=10, b=0.03170, c=0.07 * np.pi, n_var=n_var).evaluate(
        (x + 400).reshape(-1, n_var)) * 2.3
    raw_val += get_problem("Ackley", a=10, b=0.042, c=0.178 * np.pi, n_var=n_var).evaluate(
        (x - 360).reshape(-1, n_var)) * 1.7
    # _,env = signal.gausspulse(x-10, fc=0.1, bw=0.7, retenv=True)
    # raw_val += -env*10
    raw_val += window_func(x - 13, -160, +40) * np.sin((x - 27) * np.pi * 0.1) * 7
    return raw_val


def func_type_J(x, n_var, lb=-100, ub=100):
    _a = 20
    _b = 0.1
    _c = np.pi * 0.1

    global_trend = x ** 2.0 * 0.007
    func_vals = global_trend

    lopts = [(-60, 30), (7, 5), (+60, 30)]
    bins = [lb, ] + [
        (lopts[i - 1][0] + lopts[i][0]) / 2.0
        for i in range(1, len(lopts))
    ] + [ub, ]

    for lol_i in range(0, len(lopts)):
        lol, _level = lopts[lol_i]
        local_opt = get_problem("Ackley", a=_a, b=_b, c=_c, n_var=n_var).evaluate(
            (x - lol).reshape(-1, n_var)
        )
        local_opt += gauss_pulse_noise(x, fc=0.2, bw=0.2, offset=lol) * _level
        # local_opt *= window_func(x, bins[lol_i], bins[lol_i + 1])
        func_vals += local_opt

    return func_vals


def func_type_RKHS(x, n_var, lb=0.0, ub=1.0):
    assert n_var == 1
    func_vals = np.array([rkhs_synth(_x) for _x in x.flatten()]).reshape(-1, 1)
    return func_vals


def func_type_RKHS_S(x, n_var, lb=0.0, ub=1.0):
    assert n_var == 1
    func_vals = np.array([rkhs_synth(_x) * 10.0 for _x in x.flatten()]).reshape(-1, 1)
    return func_vals


def func_type_RKHSB(x, n_var, lb=0.0, ub=1.0):
    assert n_var == 1
    # func_vals = np.array([rkhs_synth(_x) for _x in x.flatten()]).reshape(-1, 1)
    func_vals = rkhs_synth(x).reshape(-1, 1)
    bump = np.exp(-((x - 0.25) * 24) ** 2.0) * 7.2
    return func_vals + bump


def func_type_RKHST(x, n_var, lb=0.0, ub=1.0):
    assert n_var == 1
    # func_vals = np.array([rkhs_synth(_x) for _x in x.flatten()]).reshape(-1, 1)
    y1 = rkhs_synth(x).reshape(-1, 1)
    y2 = stats.truncnorm.pdf(x - 0.5, loc=0, scale=0.02, a=0.0, b=3.0) * 0.2
    y3 = -np.sinc((x - 0.48) * np.pi * 15) * 5

    y = y1 + y2 + y3

    return y


def func_type_bumped_bowl(x, n_var=2, lb=-4.0, ub=4.0):
    x_norm = np.linalg.norm(x, axis=-1)
    ret = np.log(
        x_norm ** 2.0 * 0.8
        + np.exp(-10.0 * (x_norm ** 2.0))
    ) * 2.0 + 2.54
    return ret


def func_type_bumped_bowl_HD(x, n_var, lb=-4.0, ub=4.0):
    assert n_var >= 3
    x_first_2d = x[..., :2]
    x_remaining_dim = x[:, 2:]
    g = np.sum(x_remaining_dim ** 2.0 * 5.0, axis=-1) + 1.0

    f = func_type_bumped_bowl(x_first_2d, n_var=2)
    ret = f * g
    return ret


def func_type_K(x, n_var, lb=0.0, ub=1.0):
    assert n_var == 1
    y = np.sinc(x * 2 * np.pi)
    y += np.exp(-((x - 0.25) * 37) ** 2.0) * 7.2
    y += np.exp(-((x - 0.35) * 39) ** 2.0) * 8.9
    return y


def func_bertsimas(x, n_var, lb=(-0.95, -0.45), ucb=(3.2, 4.4)):
    assert n_var == 2 and x.shape[-1] == 2
    x1, x2 = x[:, 0], x[:, 1]
    ret = -2.0 * (x1 ** 6.0) + 12.2 * (x1 ** 5.0) - 21.2 * (x1 ** 4.0) + 6.4 * (x1 ** 3.0) \
          + 4.7 * (x1 ** 2.0) - 6.2 * x1 - x2 ** 6 + 11.0 * (x2 ** 5.0) - 43.3 * (x2 ** 4.0) \
          + 74.8 * (x2 ** 3.0) - 56.9 * (x2 ** 2.0) + 10 * x2 + 4.1 * x1 * x2 \
          + 0.1 * (x1 ** 2.0) * (x2 ** 2.0) - 0.4 * x1 * (x2 ** 2.0) - 0.4 * (x1 ** 2.0) * x2
    return ret


def window_func(x, start, end):
    return np.where(np.logical_and(x >= start, x <= end), 1, 0)


def get_test_problem(raw_func_name, n_var=1, raw_func_kwargs=None,
                     x_offset=3, input_range_offset=0,
                     crpt_method='raw', mesh_sample_num=500,
                     **kwargs):
    """
    Get a problem
    :param raw_func_name: raw function name
    :param n_var: number of variables
    :param raw_func_kwargs: kwargs for raw function
    :param crpt_method: corruption method, can be ['raw', 'small', 'large']
    :param mesh_sample_num: number of points for mesh generation
    :param kwargs: additional parameters to the problem setup
    :return: problem instance
    """
    f_min = None
    f_max = None
    minimization = True
    raw_parallel_eval = True
    kwargs = {}
    if raw_func_name == "Ackley":
        if raw_func_kwargs is None:
            raw_func_kwargs = {}
        raw_func = get_problem(raw_func_name, n_var=n_var, **raw_func_kwargs).evaluate
        input_bounds = kwargs.get('input_bounds', [(-40, +40), ] * n_var)
        f_max = 25.0
        f_min = 0.0

    elif raw_func_name == "Sin":
        freq = kwargs.get('freq', 0.01)
        amplitude = kwargs.get('amplitude', 10.0)
        raw_func = lambda x: amplitude * np.sin(2 * np.pi * freq * x)
        input_bounds = kwargs.get('input_bounds', [(-100, +95), ] * n_var)
        f_max = 1.0
        f_min = 0.0

    elif raw_func_name == "Griewank":
        if raw_func_kwargs is None:
            raw_func_kwargs = {}
        raw_func = get_problem(raw_func_name, n_var=n_var, **raw_func_kwargs).evaluate
        input_bounds = kwargs.get('input_bounds', [(-200, +200), ] * n_var)
        f_max = 200.0
        f_min = 0.0

    elif raw_func_name == "Rastrigin":
        if raw_func_kwargs is None:
            raw_func_kwargs = {}
        raw_func = get_problem(raw_func_name, n_var=n_var, **raw_func_kwargs).evaluate
        input_bounds = kwargs.get('input_bounds', [(-6, +6), ] * n_var)
        # f_max = 90.0
        # f_min = 0.0

    elif raw_func_name == "Rosenbrock":
        if raw_func_kwargs is None:
            raw_func_kwargs = {}
        raw_func = get_problem(raw_func_name, n_var=n_var, **raw_func_kwargs).evaluate
        input_bounds = kwargs.get('input_bounds', [(-5, +10), ] * n_var)
        f_min = 0.0
        f_max = 1.6e5

    elif raw_func_name == "Schwefel":
        if raw_func_kwargs is None:
            raw_func_kwargs = {}
        raw_func = get_problem(raw_func_name, n_var=n_var, **raw_func_kwargs).evaluate
        input_bounds = kwargs.get('input_bounds', [(-500, +500), ] * n_var)
        f_min = 0.0
        f_max = 1600.0

    elif raw_func_name == "Sphere":
        if raw_func_kwargs is None:
            raw_func_kwargs = {}
        raw_func = get_problem(raw_func_name, n_var=n_var, **raw_func_kwargs).evaluate
        input_bounds = kwargs.get('input_bounds', [(-6, +6), ] * n_var)
        # f_min = 0.0
        # f_max = 80.0

    elif raw_func_name == "Zakharov":
        if raw_func_kwargs is None:
            raw_func_kwargs = {}
        raw_func = get_problem(raw_func_name, n_var=n_var, **raw_func_kwargs).evaluate
        input_bounds = kwargs.get('input_bounds', [(-5, +10), ] * n_var)
        f_min = 0.0
        f_max = 6e4

    elif raw_func_name == "Sinc":
        assert n_var == 1
        input_bounds = kwargs.get('input_bounds', [(-3 * math.pi, +3 * math.pi), ])
        raw_func = lambda x: -np.sinc(x)
        f_min = -0.25
        f_max = 1.0

    elif raw_func_name == "Sin":
        assert n_var == 1
        input_bounds = kwargs.get('input_bounds', [(-3 * math.pi, +3 * math.pi), ])
        raw_func = lambda x: -np.sin(x)
        f_min = -1.0
        f_max = 1.0

    # elif raw_func_name == "CustomA":
    #     if raw_func_kwargs is None:
    #         raw_func_kwargs = {}
    #     input_bounds = kwargs.get('input_bounds', [(-500, +500), ] * n_var)
    #     _f1 = get_problem('schwefel', n_var=n_var, **raw_func_kwargs).evaluate
    #     raw_func = lambda x: (abs(x) * 1.2) * _f1(x)
    #     f_min = 0.0
    #     f_max = 1600.0

    elif raw_func_name == "CustomB":
        raw_func = lambda x: (np.sign(x) * 2) + np.sin(10 * x)
        input_bounds = kwargs.get('input_bounds', [(-3 * np.pi, +3 * np.pi), ] * n_var)
        f_min = -3.0
        f_max = 3.0

    elif raw_func_name == "CustomC":
        raw_func = partial(func_type_C, n_var=n_var, with_noise=True)
        input_bounds = kwargs.get('input_bounds', [(-50, +45), ] * n_var)
        f_max = 40.0
        f_min = -15.0

    elif raw_func_name == "CustomD":
        level = kwargs.get('level', 4)
        raw_func = partial(func_type_D, n_var=n_var, level=level)
        input_bounds = kwargs.get('input_bounds', [(-50, +45), ] * n_var)

    elif raw_func_name == "CustomE":
        input_bounds = kwargs.get('input_bounds', [(-50, +45), ] * n_var)
        raw_func = partial(func_type_E, n_var=n_var, xl=input_bounds[0][0], xu=input_bounds[0][1])

    elif raw_func_name == "CustomE1":
        input_bounds = kwargs.get('input_bounds', [(-100, +100), ] * n_var)
        raw_func = partial(func_type_E1, n_var=n_var, xl=input_bounds[0][0], xu=input_bounds[0][1])


    elif raw_func_name == "CustomF":
        input_bounds = kwargs.get('input_bounds', [(-80, +75), ] * n_var)
        raw_func = partial(func_type_F, n_var=n_var, xl=input_bounds[0][0], xu=input_bounds[0][1])

    elif raw_func_name == "CustomG":
        input_bounds = kwargs.get('input_bounds', [(-50, +45), ] * n_var)
        raw_func = partial(func_type_G, n_var=n_var, xl=input_bounds[0][0], xu=input_bounds[0][1])

    elif raw_func_name == "CustomH":
        raw_func = partial(func_type_H, n_var=n_var, with_noise=True)
        input_bounds = kwargs.get('input_bounds', [(-100, +95), ] * n_var)

    elif raw_func_name == "CustomH1":
        raw_func = partial(func_type_H1, n_var=n_var, with_noise=True)
        input_bounds = kwargs.get('input_bounds', [(-100, +95), ] * n_var)
    elif raw_func_name == "CustomI":
        raw_func = partial(func_type_I, n_var=n_var, with_noise=True)
        input_bounds = kwargs.get('input_bounds', [(-500, +500), ] * n_var)
    elif raw_func_name == "CustomJ":
        raw_func = partial(func_type_J, n_var=n_var, lb=-100, ub=+100)
        input_bounds = kwargs.get('input_bounds', [(-100, +100), ] * n_var)
    elif raw_func_name == "RKHS":
        raw_func = partial(func_type_RKHS, n_var=n_var, lb=0.0, ub=+1.0)
        input_bounds = kwargs.get('input_bounds', [(0.0, 1.0), ] * n_var)
        minimization = False
    elif raw_func_name == "RKHS-S":
        raw_func = partial(func_type_RKHS_S, n_var=n_var, lb=0.0, ub=+1.0)
        input_bounds = kwargs.get('input_bounds', [(0.0, 1.0), ] * n_var)
        minimization = False
    elif raw_func_name == "RKHS-W":
        raw_func = partial(func_type_RKHS, n_var=n_var, lb=0.0, ub=+1.0)
        input_bounds = kwargs.get('input_bounds', [(-1.0, 1.0), ] * n_var)
        minimization = False
    elif raw_func_name == "RKHSB":
        raw_func = partial(func_type_RKHSB, n_var=n_var, lb=0.0, ub=+1.0)
        input_bounds = kwargs.get('input_bounds', [(0.0, 1.0), ] * n_var)
        minimization = False
    elif raw_func_name == "RKHST":
        raw_func = partial(func_type_RKHST, n_var=n_var, lb=0.0, ub=+1.0)
        input_bounds = kwargs.get('input_bounds', [(0.0, 1.0), ] * n_var)
        minimization = False
    elif raw_func_name == "CustomK":
        raw_func = partial(func_type_K, n_var=n_var, lb=0.0, ub=+1.0)
        input_bounds = kwargs.get('input_bounds', [(0.0, 1.0), ] * n_var)
        minimization = False
    elif raw_func_name == "BumpedBowl":
        raw_func = partial(func_type_bumped_bowl, n_var=n_var, lb=-2.0, ub=+2.0)
        input_bounds = kwargs.get('input_bounds', [(-2.0, 2.0), ] * n_var)
        minimization = True
    elif raw_func_name == "BumpedBowlHD":
        raw_func = partial(func_type_bumped_bowl_HD, n_var=n_var, lb=-2.0, ub=+2.0)
        input_bounds = kwargs.get('input_bounds', [(-2.0, 2.0), ] * n_var)
        minimization = True
    elif raw_func_name == "Bertsimas":
        raw_func = partial(func_bertsimas, n_var=n_var)
        input_bounds = kwargs.get('input_bounds', [(-0.95, 3.2), (-0.45, 4.4)])
        minimization = False
    elif raw_func_name == "Push3":
        assert n_var == 3
        from problems.robot_pushing import push_env as pe
        anchor_goal = np.array((-3, 3))
        env = pe.Push3Env(anchor_goal=anchor_goal)
        raw_func = env.evaluate
        input_bounds = kwargs.get('input_bounds', [(-5, 5), (-5, 5), (1, 30)])
        minimization = True
        raw_parallel_eval = False
        f_min = 0
        f_max = np.linalg.norm(
            np.array([input_bounds[0][0], input_bounds[1][1]])
            - np.array([input_bounds[0][1], input_bounds[1][0]]))
        kwargs = {'env': env}
    elif raw_func_name == "DualGoalsP3":
        assert n_var == 3
        from problems.robot_pushing import push_env as pe
        anchor_goal = np.array((-3, 3))
        env = pe.DualGoalsP3Env(anchor_goal=anchor_goal)
        raw_func = env.evaluate
        input_bounds = kwargs.get('input_bounds', [(-5, 5), (-5, 5), (1, 30)])
        minimization = True
        raw_parallel_eval = False
        f_min = 0
        f_max = np.linalg.norm(
            np.array([input_bounds[0][0], input_bounds[1][1]])
            - np.array([input_bounds[0][1], input_bounds[1][0]]))
        kwargs = {'env': env}
    elif raw_func_name == "TripleGoalsP3":
        assert n_var == 3
        from problems.robot_pushing import push_env as pe
        env = pe.TripleGoalsP3Env(show_gui=False)
        raw_func = env.evaluate
        input_bounds = kwargs.get('input_bounds', [(-5, 5), (-5, 5), (1, 30)])
        minimization = True
        raw_parallel_eval = False
        f_min = 0
        f_max = np.linalg.norm(
            np.array([input_bounds[0][0], input_bounds[1][1]])
            - np.array([input_bounds[0][1], input_bounds[1][0]]))
        kwargs = {'env': env}
    elif raw_func_name == "DualGoalsP4":
        assert n_var == 4
        from problems.robot_pushing import push_env as pe
        env = pe.DualGoalsP4Env(show_gui=False)
        raw_func = env.evaluate
        input_bounds = kwargs.get('input_bounds', [(-5, 5), (-5, 5), (1, 30),
                                                   (-np.pi * 0.5, np.pi * 0.5)])
        minimization = True
        raw_parallel_eval = False
        f_min = 0
        f_max = np.linalg.norm(
            np.array([input_bounds[0][0], input_bounds[1][1]])
            - np.array([input_bounds[0][1], input_bounds[1][0]]))
        kwargs = {'env': env}
    elif raw_func_name == "TripleGoalsP4":
        assert n_var == 4
        from problems.robot_pushing import push_env as pe
        env = pe.TripleGoalsP4Env(show_gui=False)
        raw_func = env.evaluate
        input_bounds = kwargs.get('input_bounds', [(-5, 5), (-5, 5), (1, 30),
                                                   (-np.pi * 0.5, np.pi * 0.5)])
        minimization = True
        raw_parallel_eval = False
        f_min = 0
        f_max = np.linalg.norm(
            np.array([input_bounds[0][0], input_bounds[1][1]])
            - np.array([input_bounds[0][1], input_bounds[1][0]]))
        kwargs = {'env': env}
    else:
        raise ValueError("Unsupported problem name:", raw_func_name)

    prob = TestFunctions(n_var, n_obj=1, n_constr=0,
                         xl=[input_bounds[i][0] for i in range(n_var)],
                         xu=[input_bounds[i][1] for i in range(n_var)],
                         raw_func=raw_func, raw_parallel_eval=raw_parallel_eval,
                         raw_f_min=f_min, raw_f_max=f_max,
                         crpt_method=crpt_method, mesh_granularity=mesh_sample_num,
                         input_range_offset=input_range_offset, x_offset=x_offset,
                         name=raw_func_name, minimization=minimization, **kwargs)
    return prob


if __name__ == "__main__":
    from utils import visulaization as vis
    from scipy import stats
    from model_utils.input_transform import add_noise, additional_xc_samples
    from utils import input_uncertainty as iu

    # for f_name in BENCHMARK_FUNCTION_NAMES:
    for f_name, n_var in [
        ('RKHS-S', 1),
        ('CustomK', 1),
        # ('BumpedBowl', 2),
    ]:
        prob = get_test_problem(
            f_name, n_var=n_var, crpt_method="raw", mesh_sample_num=1000,
            input_range_offset=0, x_offset=0
        )
        num_expectation_eval = 50 * (2 ** (n_var - 1))
        prob.evaluate(np.random.rand(7, 100))

        if prob.n_var < 3:
            # find the exact and robust optimum
            exact_opt_ind = np.argmin(prob.mesh_vals) if prob.minimization \
                else np.argmax(prob.mesh_vals)
            exact_opt_x = prob.mesh_coords[exact_opt_ind: exact_opt_ind + 1]
            exact_opt_y = prob.mesh_vals.flatten()[exact_opt_ind]

            raw_input_std = 0.1
            # input_distrib = iu.ScipyInputDistribution(
            #     stats.norm(loc=0, scale=0.25), name="Gaussian", n_var=n_var
            # )
            # input_distrib = iu.ScipyInputDistribution(
            #     stats.beta(a=0.4, b=0.2, scale=0.2, loc=-0.4), name='beta', n_var=1
            # )
            # input_distrib = iu.Circular2Distribution(
            #     stats.uniform(),
            #     name='circular2D', radius=0.5,
            # )
            input_distrib = iu.ScipyInputDistribution(
                stats.norm(loc=0.0, scale=raw_input_std) if prob.n_var == 1 \
                    else stats.multivariate_normal(mean=[0.0, 0.0], cov=raw_input_std),
                name="Gaussian", n_var=n_var
            )
            input_sampling_func = input_distrib.sample

            all_expectations = prob.evaluate(
                additional_xc_samples(
                    prob.mesh_coords, num_expectation_eval, prob.n_var, input_sampling_func
                ).reshape(-1, prob.n_var)
            ).reshape(prob.mesh_coords.shape[0], -1).mean(axis=-1)
            robust_opt_ind = np.argmin(all_expectations) if prob.minimization \
                else np.argmax(all_expectations)
            robust_opt_x = prob.mesh_coords[robust_opt_ind:robust_opt_ind + 1]
            robust_opt_y = all_expectations.flatten()[robust_opt_ind]
            print(f"{prob.name} \n"
                  f"[exact] opt_x={exact_opt_x}, opt_y={exact_opt_y}, "
                  f"E[y]={all_expectations[exact_opt_ind]}. \n"
                  f"[robust] opt_x={robust_opt_x}, opt_y={robust_opt_y}")

            fig, axes = vis.visualize_problem(
                prob, plot_corrupted=False,
                anotations={
                    'exact_opt': (exact_opt_x, exact_opt_y,
                                  {'marker': 'o', 'color': 'r', 's': 50, 'alpha': 0.7}),
                    'robust_opt': (robust_opt_x, prob.mesh_vals[robust_opt_ind],
                                   {'marker': '*', 'color': 'g', 's': 50, 'alpha': 0.7}),
                }
            )
            # plt.plot(prob.mesh_coords, all_expectations, alpha=0.6, ls="-.", c='b')
            if n_var == 1:
                plt.axvline(exact_opt_x, c='r', ls='--', alpha=0.6, lw=1, zorder=100)
                plt.axvline(robust_opt_x, c='g', ls='--', alpha=0.6, lw=1, zorder=100)
            fig.legend()
            plt.show()
        else:
            print("problem of n_var > 2, cannot visualize")

        if (prob.mesh_vals.min() >= prob.min_value) \
                and (prob.mesh_vals.max() <= prob.max_value):
            pass
        else:
            print(f"[{f_name}] {prob.mesh_vals.min():.3f}, {prob.min_value:.3f}, "
                  f"{prob.mesh_vals.max():.3f}, {prob.max_value:.3f}")

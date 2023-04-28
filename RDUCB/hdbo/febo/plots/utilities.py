"""
Utilities for the safeopt library (e.g., sampling).

"""

from __future__ import print_function, absolute_import, division

from collections import Sequence            # isinstance(...,Sequence)
import numpy as np
import scipy as sp

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D     # Create 3D axes
    from matplotlib import cm                   # 3D plot colors
except ImportError:
    print("Plotting library could not be loaded.")


__all__ = ['linearly_spaced_combinations', 'sample_gp_function', 'plot_2d_gp',
           'plot_3d_gp', 'plot_contour_gp']


def linearly_spaced_combinations(bounds, num_samples):
    """
    Return 2-D array with all linearly spaced combinations with the bounds.

    Parameters
    ----------
    bounds: sequence of tuples
        The bounds for the variables, [(x1_min, x1_max), (x2_min, x2_max), ...]
    num_samples: integer or array_likem
        Number of samples to use for every dimension. Can be a constant if
        the same number should be used for all, or an array to fine-tune
        precision. Total number of data points is num_samples ** len(bounds).

    Returns
    -------
    combinations: 2-d array
        A 2-d arrray. If d = len(bounds) and l = prod(num_samples) then it
        is of size l x d, that is, every row contains one combination of
        inputs.
    """
    num_vars = len(bounds)

    if not isinstance(num_samples, Sequence):
        num_samples = [num_samples] * num_vars

    if len(bounds) == 1:
        return np.linspace(bounds[0][0], bounds[0][1], num_samples[0])[:, None]

    # Create linearly spaced test inputs
    inputs = [np.linspace(b[0], b[1], n) for b, n in zip(bounds,
                                                         num_samples)]

    # Convert to 2-D array
    return np.array([x.ravel() for x in np.meshgrid(*inputs)]).T


def sample_gp_function(kernel, bounds, noise_var, num_samples,
                       interpolation='kernel', mean_function=None):
    """
    Sample a function from a gp with corresponding kernel within its bounds.

    Parameters
    ----------
    kernel: instance of GPy.kern.*
    bounds: list of tuples
        [(x1_min, x1_max), (x2_min, x2_max), ...]
    noise_var: float
        Variance of the observation noise of the GP function
    num_samples: int or list
        If integer draws the corresponding number of samples in all
        dimensions and test all possible input combinations. If a list then
        the list entries correspond to the number of linearly spaced samples of
        the corresponding input
    interpolation: string
        If 'linear' interpolate linearly between samples, if 'kernel' use the
        corresponding mean RKHS-function of the GP.
    mean_function: callable
        Mean of the sample function

    Returns
    -------
    function: object
        function(x, noise=True)
        A function that takes as inputs new locations x to be evaluated and
        returns the corresponding noisy function values. If noise=False is
        set the true function values are returned (useful for plotting).
    """
    inputs = linearly_spaced_combinations(bounds, num_samples)
    cov = kernel.K(inputs) + np.eye(inputs.shape[0]) * 1e-6
    output = np.random.multivariate_normal(np.zeros(inputs.shape[0]),
                                           cov)

    if interpolation == 'linear':

        def evaluate_gp_function_linear(x, noise=True):
            """
            Evaluate the GP sample function with linear interpolation.

            Parameters
            ----------
            x: np.array
                2D array with inputs
            noise: bool
                Whether to include prediction noise
            """
            x = np.atleast_2d(x)
            y = sp.interpolate.griddata(inputs, output, x, method='linear')

            # Work around weird dimension squishing in griddata
            y = np.atleast_2d(y.squeeze()).T

            if mean_function is not None:
                y += mean_function(x)
            if noise:
                y += np.sqrt(noise_var) * np.random.randn(x.shape[0], 1)
            return y
        return evaluate_gp_function_linear

    elif interpolation == 'kernel':
        cho_factor = sp.linalg.cho_factor(cov)
        alpha = sp.linalg.cho_solve(cho_factor, output)

        def evaluate_gp_function_kernel(x, noise=True):
            """
            Evaluate the GP sample function with kernel interpolation.

            Parameters
            ----------
            x: np.array
                2D array with inputs
            noise: bool
                Whether to include prediction noise
            """
            x = np.atleast_2d(x)
            y = kernel.K(x, inputs).dot(alpha)
            y = y[:, None]
            if mean_function is not None:
                y += mean_function(x)
            if noise:
                y += np.sqrt(noise_var) * np.random.randn(x.shape[0], 1)
            return y

        return evaluate_gp_function_kernel


def plot_2d_gp(gp, inputs, predictions=None, figure=None, axis=None,
               fixed_inputs=None, beta=3, fmin=None, **kwargs):
        """
        Plot a 2D GP with uncertainty.

        Parameters
        ----------
        gp: Instance of GPy.models.GPRegression
        inputs: 2darray
            The input parameters at which the GP is to be evaluated
        predictions: ndarray
            Can be used to manually pass the GP predictions, set to None to
            use the gp directly. Is of the form (mean, variance)
        figure: matplotlib figure
            The figure on which to draw (ignored if axis is provided
        axis: matplotlib axis
            The axis on which to draw
        fixed_inputs: list
            A list containing the the fixed inputs and their corresponding
            values, e.g., [(0, 3.2), (4, -2.43)]. Set the value to None if
            it's not fixed, but should not be a plotted axis either
        beta: float
            The confidence interval used
        fmin : float
            The safety threshold value.

        Returns
        -------
        axis
        """
        if fixed_inputs is None:
            if gp.kern.input_dim > 1:
                raise NotImplementedError('This only works for 1D inputs')
            fixed_inputs = []
        elif gp.kern.input_dim - len(fixed_inputs) != 1:
            raise NotImplemented('This only works for 1D inputs')

        ms = kwargs.pop('ms', 10)
        mew = kwargs.pop('mew', 3)
        point_color = kwargs.pop('point_color', 'k')

        if axis is None:
            if figure is None:
                figure = plt.figure()
                axis = figure.gca()
            else:
                axis = figure.gca()

        # Get a list of unfixed inputs to plot
        unfixed = list(range(gp.kern.input_dim))
        for dim, val in fixed_inputs:
            if val is not None:
                inputs[:, dim] = val
            unfixed.remove(dim)

        # Compute GP predictions if not provided
        if predictions is None:
            mean, var = gp._raw_predict(inputs)
        else:
            mean, var = predictions

        output = mean.squeeze()
        std_dev = beta * np.sqrt(var.squeeze())

        axis.fill_between(inputs[:, unfixed[0]],
                          output - std_dev,
                          output + std_dev,
                          facecolor='blue',
                          alpha=0.3)

        axis.plot(inputs[:, unfixed[0]], output, **kwargs)
        axis.scatter(gp.X[:-1, unfixed[0]], gp.Y[:-1, 0], s=20 * ms,
                     marker='x', linewidths=mew, color=point_color)
        axis.scatter(gp.X[-1, unfixed[0]], gp.Y[-1, 0], s=20 * ms,
                     marker='x', linewidths=mew, color='r')
        axis.set_xlim([np.min(inputs[:, unfixed[0]]),
                       np.max(inputs[:, unfixed[0]])])

        if fmin is not None:
            axis.plot(inputs[[0, -1], unfixed[0]], [fmin, fmin], 'k--')

        return axis


def plot_3d_gp(gp, inputs, predictions=None, figure=None, axis=None,
               fixed_inputs=None, beta=3, **kwargs):
        """
        Plot a 3D gp with uncertainty.

        Parameters
        ----------
        gp: Instance of GPy.models.GPRegression
        inputs: 2darray
            The input parameters at which the GP is to be evaluated
        predictions: ndarray
            Can be used to manually pass the GP predictions, set to None to
            use the gp directly. Is of the form [mean, variance]
        figure: matplotlib figure
            The figure on which to draw (ignored if axis is provided
        axis: matplotlib axis
            The axis on which to draw
        fixed_inputs: list
            A list containing the the fixed inputs and their corresponding
            values, e.g., [(0, 3.2), (4, -2.43)]. Set the value to None if
            it's not fixed, but should not be a plotted axis either
        beta: float
            The confidence interval used

        Returns
        -------
        surface: matplotlib trisurf plot
        data: matplotlib plot for data points
        """
        if fixed_inputs is None:
            if gp.kern.input_dim > 2:
                raise NotImplementedError('This only works for 2D inputs')
            fixed_inputs = []
        elif gp.kern.input_dim - len(fixed_inputs) != 2:
            raise NotImplemented('Only two inputs can be unfixed')

        if axis is None:
            if figure is None:
                figure = plt.figure()
                axis = Axes3D(figure)
            else:
                axis = Axes3D(figure)

        # Get a list of unfixed inputs to plot
        unfixed = list(range(gp.kern.input_dim))
        for dim, val in fixed_inputs:
            if val is not None:
                inputs[:, dim] = val
            unfixed.remove(dim)

        # Compute GP predictions if not provided
        if predictions is None:
            mean, var = gp._raw_predict(inputs)
        else:
            mean, var = predictions

        surf = axis.plot_trisurf(inputs[:, unfixed[0]],
                                 inputs[:, unfixed[1]],
                                 mean[:, 0],
                                 cmap=cm.jet, linewidth=0.2, alpha=0.5)

        data = axis.plot(gp.X[:-1, unfixed[0]],
                         gp.X[:-1, unfixed[1]],
                         gp.Y[:-1, 0],
                         'o')
        axis.plot(gp.X[-1, unfixed[0]],
                  gp.X[-1, unfixed[1]],
                  gp.Y[-1, 0],
                  'ro')

        axis.set_xlim([np.min(inputs[:, unfixed[0]]),
                       np.max(inputs[:, unfixed[0]])])

        axis.set_ylim([np.min(inputs[:, unfixed[1]]),
                       np.max(inputs[:, unfixed[1]])])

        return surf, data

def plot_x_and_f(f, inputs, X,  axis=None, figure=None):
    if axis is None:
        if figure is None:
            figure = plt.figure()
            axis = figure.gca()
        else:
            axis = figure.gca()

    slices = []
    lengths = []
    for i, inp in enumerate(inputs):
        if isinstance(inp, np.ndarray):
            slices.append(i)
            lengths.append(inp.shape[0])

    axis.set_xlim([np.min(inputs[slices[0]]),
                   np.max(inputs[slices[0]])])

    axis.set_ylim([np.min(inputs[slices[1]]),
                   np.max(inputs[slices[1]])])


    data = axis.plot(X[:, slices[0]], X[:, slices[1]], 'xk', markersize=15)

def plot_contour_gp(gp, inputs, predictions=None, figure=None, axis=None,
                    colorbar=True, red_points=None, green_points=None, blue_points=None, **kwargs):
        """
        Plot a 3D gp with uncertainty.

        Parameters
        ----------
        gp: Instance of GPy.models.GPRegression
        inputs: list of arrays/floats
            The input parameters at which the GP is to be evaluated,
            here instead of the combinations of inputs the individual inputs
            that are spread in a grid are given. Only two of the arrays
            should have more than one value (not fixed).
        predictions: ndarray
            Can be used to manually pass the GP predictions, set to None to
            use the gp directly.
        figure: matplotlib figure
            The figure on which to draw (ignored if axis is provided
        axis: matplotlib axis
            The axis on which to draw

        Returns
        -------
        contour: matplotlib contour plot
        colorbar: matplotlib colorbar
        points: matplotlib plot
        """
        if axis is None:
            if figure is None:
                figure = plt.figure()
                axis = figure.gca()
            else:
                axis = figure.gca()

        if green_points is not None:
            data = axis.plot(green_points[:, slices[0]], green_points[:, slices[1]], 'og')
        if red_points is not None:
            data = axis.plot(red_points[:, slices[0]], red_points[:, slices[1]], 'or')
        if blue_points is not None:
            data = axis.plot(blue_points[:, slices[0]], blue_points[:, slices[1]], 'ob')

        # Find which inputs are fixed to constant values
        slices = []
        lengths = []
        for i, inp in enumerate(inputs):
            if isinstance(inp, np.ndarray):
                slices.append(i)
                lengths.append(inp.shape[0])


        mesh = np.meshgrid(*inputs, indexing='ij')
        if predictions is None:
            # Convert to array with combinations of inputs
            gp_inputs = np.array([x.ravel() for x in mesh]).T
            mean = gp.mean_var(gp_inputs)[0]
        else:
            mean = predictions[0]

        c_bar = None
        if not np.all(mean == mean[0]):
            # Need to squeeze the added dimensions caused by fixed inputs
            c = axis.contour(mesh[slices[0]].squeeze(),
                             mesh[slices[1]].squeeze(),
                             mean.squeeze().reshape(*lengths),
                             20,
                             **kwargs)
            if colorbar:
                c_bar = plt.colorbar(c)
        else:
            c = None



        data = axis.plot(gp._X[:, slices[0]], gp._X[:, slices[1]], 'xk', markersize=15)
        # axis.plot(gp.X[-1, slices[0]], gp.X[-1, slices[1]], 'or')

        axis.set_xlim([np.min(inputs[slices[0]]),
                       np.max(inputs[slices[0]])])

        axis.set_ylim([np.min(inputs[slices[1]]),
                       np.max(inputs[slices[1]])])

        return c, c_bar, data


def plot_safeset(safeset_function, inputs, predictions=None, figure=None, axis=None,
                    colorbar=True, **kwargs):
    """
    Plot a 3D gp with uncertainty.

    Parameters
    ----------
    gp: Instance of GPy.models.GPRegression
    inputs: list of arrays/floats
        The input parameters at which the GP is to be evaluated,
        here instead of the combinations of inputs the individual inputs
        that are spread in a grid are given. Only two of the arrays
        should have more than one value (not fixed).
    predictions: ndarray
        Can be used to manually pass the GP predictions, set to None to
        use the gp directly.
    figure: matplotlib figure
        The figure on which to draw (ignored if axis is provided
    axis: matplotlib axis
        The axis on which to draw

    Returns
    -------
    contour: matplotlib contour plot
    colorbar: matplotlib colorbar
    points: matplotlib plot
    """
    if axis is None:
        if figure is None:
            figure = plt.figure()
            axis = figure.gca()
        else:
            axis = figure.gca()


    # Find which inputs are fixed to constant values
    slices = []
    lengths = []
    for i, inp in enumerate(inputs):
        if isinstance(inp, np.ndarray):
            slices.append(i)
            lengths.append(inp.shape[0])

    mesh = np.meshgrid(*inputs, indexing='ij')
    gp_inputs = np.array([x.ravel() for x in mesh]).T
    mean = safeset_function(gp_inputs)

    if not np.all(mean == mean[0]):
        # Need to squeeze the added dimensions caused by fixed inputs
        c = axis.contour(mesh[slices[0]].squeeze(),
                         mesh[slices[1]].squeeze(),
                         mean.reshape(*lengths),
                         levels=[0],
                         colors=['red'],
                         linewidths=2,
                         linestyles='dashed',
                         **kwargs)
    else:
        c = None


    axis.set_xlim([np.min(inputs[slices[0]]),
                   np.max(inputs[slices[0]])])

    axis.set_ylim([np.min(inputs[slices[1]]),
                   np.max(inputs[slices[1]])])

    return c


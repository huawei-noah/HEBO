""" Code to do Bayesian Optimization with GP """

import argparse
import logging
import functools
import sys
import time
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow

# configs
gpflow.config.set_default_float(np.float32)

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--logfile",
    type=str,
    help="file to log to",
    default="gp_opt.log"
)
parser.add_argument(
    "--seed",
    type=int,
    required=True
)
parser.add_argument(
    "--gp_file",
    type=str,
    required=True,
    help="file to load GP hyperparameters from",
)
parser.add_argument(
    "--gp_err_file",
    type=str,
    help="file to load error GP hyperparameters from",
    default=None,
)
parser.add_argument(
    "--data_file",
    type=str,
    help="file to load data from",
    required=True
)
parser.add_argument(
    "--data_err_file",
    type=str,
    help="file to load reconstruction error data from",
    default=None
)
parser.add_argument(
    "--save_file",
    type=str,
    required=True,
    help="File to save results to"
)
parser.add_argument(
    "--n_out",
    type=int,
    default=1,
    help="Number of optimization points to return"
)
parser.add_argument(
    "--n_starts",
    type=int,
    default=20,
    help="Number of optimization starts to use"
)
parser.add_argument(
    "--no_early_stopping",
    dest="early_stopping",
    action="store_false",
    help="Flag to turn off early stopping"
)


# Functions to calculate expected improvement
# =============================================================================
def _ei_tensor(x):
    """ convert arguments to tensor for ei calcs """
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    return tf.convert_to_tensor(x, dtype=tf.float32)


def neg_ei(x, gp, fmin, check_type=True):
    if check_type:
        x = _ei_tensor(x)

    std_normal = tfp.distributions.Normal(loc=0., scale=1.)
    mu, var = gp.predict_f(x)
    sigma = tf.sqrt(var)
    z = (fmin - mu) / sigma

    ei = ((fmin - mu) * std_normal.cdf(z) +
          sigma * std_normal.prob(z))
    return -ei


def neg_eaei(x, gp_f, gp_r, fmin, eps=10., check_type=True, n_err_samples=512, input_bounds=None,
             err_input_bounds=None, target_mean=None, target_std=None, err_target_mean=None, err_target_std=None,
             var_bounds=None, err_var_bounds=None):
    def normalize(x, bounds):
        return (x - bounds[0]) / (bounds[1] - bounds[0])

    if check_type:
        x = _ei_tensor(x)

    std_normal = tfp.distributions.Normal(loc=0., scale=1.)
    mu_f, var_f = gp_f.predict_f(x)
    sigma_f = tf.sqrt(var_f)
    z = (fmin - mu_f) / sigma_f

    ei = ((fmin - mu_f) * std_normal.cdf(z) +
          sigma_f * std_normal.prob(z))

    # normalise inputs for error GP
    x_norm = normalize(x, err_input_bounds)

    # error-GP predicts standardised targets
    mu_r, var_r = gp_r.predict_f(x_norm)
    sigma_r = tf.sqrt(var_r)
    err_post = tfp.distributions.Normal(loc=mu_r, scale=sigma_r)
    err_samples = err_post.sample(n_err_samples)
    err_samples_dstd = err_samples * err_target_std + err_target_mean  # de-standardise samples
    err_samples_pos = tf.clip_by_value(err_samples_dstd, 1e-10, 1e+10)
    # normalise error predictions to avoid over/under flow
    err_samples_pos_min = tf.reduce_min(err_samples_pos)
    err_samples_pos_max = tf.reduce_max(err_samples_pos)
    err_samples_norm = (err_samples_pos - err_samples_pos_min) / (err_samples_pos_max - err_samples_pos_min + 1e-6)

    var_f_normalised = normalize(tf.stop_gradient(var_f), var_bounds) + 1e-3
    var_r_normalised = normalize(tf.stop_gradient(var_r), err_var_bounds) + 1e-3
    # gamma = tf.clip_by_value(var_r_normalised, 1e-3, 1.0) / var_f_normalised  # avoid exploding var_r
    gamma = var_r_normalised / var_f_normalised
    gamma = tf.clip_by_value(gamma, 1e-3, 10.0)  # avoid exploding var_r

    error_term = tf.pow(err_samples_norm * eps + 1., gamma)
    eaei = ei / error_term

    return tf.reduce_mean(-eaei, axis=0)


def neg_ei_and_grad(x, gp, fmin, numpy=True):
    x = _ei_tensor(x)
    with tf.GradientTape() as tape:
        tape.watch(x)
        val = neg_ei(x, gp, fmin, check_type=False)
    grad = tape.gradient(val, x)
    if numpy:
        return val.numpy(), grad.numpy()
    else:
        return val, grad


def neg_eaei_and_grad(x, gp_f, gp_r, fmin, input_bounds=None, err_input_bounds=None, target_mean=None,
                      target_std=None, err_target_mean=None, err_target_std=None, var_bounds=None,
                      err_var_bounds=None, eps=10., numpy=True):
    x = _ei_tensor(x)
    with tf.GradientTape() as tape:
        tape.watch(x)
        val = neg_eaei(x, gp_f, gp_r, fmin, eps=eps, check_type=False, input_bounds=input_bounds,
                       err_input_bounds=err_input_bounds, target_mean=target_mean, target_std=target_std,
                       err_target_mean=err_target_mean, err_target_std=err_target_std,
                       var_bounds=var_bounds, err_var_bounds=err_var_bounds)
    grad = tape.gradient(val, x)
    if numpy:
        return val.numpy(), grad.numpy()
    else:
        return val, grad


def robust_multi_restart_optimizer(
        func_with_grad,
        X_train, y_train,
        num_pts_to_return=1,
        num_random_starts=5,
        num_good_starts=5,
        good_point_cutoff=0.0,
        use_tqdm=False,
        bounds_abs=4.,
        return_res=False,
        logger=None,
        early_stop=True,
        shuffle_points=True
):
    """
    Wrapper that calls scipy's optimize function at many different start points.
    It uses a mix of random starting points, and good points in the dataset.
    """

    # wrapper for tensorflow functions, that handles array flattening and dtype changing
    def objective1d(v):
        return tuple([arr.ravel().astype(np.float64) for arr in func_with_grad(v)])

    # Set up points to optimize in
    rand_points = [np.random.randn(X_train.shape[1]).astype(np.float32)
                   for _ in range(num_random_starts)]
    top_point_idxs = np.arange(len(y_train))[(
            y_train <= good_point_cutoff).ravel()]
    chosen_top_point_indices = np.random.choice(
        top_point_idxs, size=num_good_starts, replace=False)
    top_points = [X_train[i].ravel().copy() for i in chosen_top_point_indices]
    all_points = rand_points + top_points
    point_sources = ["rand"] * len(rand_points) + ["good"] * len(top_points)

    # Optionally shuffle points (early stopping means order can matter)
    if shuffle_points:
        _list_together = list(zip(all_points, point_sources))
        np.random.shuffle(_list_together)
        all_points, point_sources = list(zip(*_list_together))
        del _list_together

    # Main optimization loop
    start_time = time.time()
    num_good_results = 0
    if use_tqdm:
        all_points = tqdm(all_points)
    opt_results = []
    for i, (x, src) in enumerate(zip(all_points, point_sources)):
        res = minimize(
            fun=objective1d, x0=x,
            jac=True,
            bounds=[(-bounds_abs, bounds_abs) for _ in range(X_train.shape[1])])

        opt_results.append(res)

        if logger is not None:
            logger.info(
                f"Iter#{i} t={time.time() - start_time:.1f}s: val={sum(res.fun):.2e}, "
                f"init={src}, success={res.success}, msg={str(res.message.decode())}")

        # Potentially do early stopping
        # Good results succeed, and stop due to convergences, not low gradients
        result_is_good = res.success and ("REL_REDUCTION_OF_F_<=_FACTR*EPSMCH" in res.message.decode())
        if result_is_good:
            num_good_results += 1
            if (num_good_results >= num_pts_to_return) and early_stop:
                logger.info(f"Early stopping since {num_good_results} good points found.")
                break

    # Potentially directly return optimization results
    if return_res:
        return opt_results

    # Find the best successful results
    successful_results = [res for res in opt_results if res.success]
    sorted_results = sorted(successful_results, key=lambda r: r.fun.sum())
    x_out = [res.x for res in sorted_results[:num_pts_to_return]]
    opt_vals_out = [res.fun.sum()
                    for res in sorted_results[:num_pts_to_return]]
    return np.array(x_out), opt_vals_out


def gp_opt(gp_file, data_file, save_file, n_out, logfile,
           gp_err_file=None, data_err_file=None,
           n_starts=20, early_stopping=True):
    """ Do optimization via GPFlow"""
    # check if using error-aware acquisition
    if gp_err_file is not None and data_err_file is not None:
        error_aware_acqf = True
    elif (gp_err_file is None and data_err_file is not None) or (gp_err_file is not None and data_err_file is None):
        raise AssertionError("either both gp_err_file and data_err_file need to be set or none of them.")
    else:
        error_aware_acqf = False

    # Set up logger
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(logging.FileHandler(logfile))
    LOGGER.addHandler(logging.StreamHandler(sys.stdout))
    # if not LOGGER.hasHandlers():
    #     LOGGER.addHandler(logging.StreamHandler(sys.stdout))

    # Load the data
    with np.load(data_file, allow_pickle=True) as npz:
        X_train = npz['X_train']
        X_test = npz['X_test']
        y_train = npz['y_train']
        y_test = npz['y_test']

    # Initialize the GP
    with np.load(gp_file, allow_pickle=True) as npz:
        Z = npz['Z']
        kernel_lengthscales = npz['kernel_lengthscales']
        kernel_variance = npz['kernel_variance']
        likelihood_variance = npz['likelihood_variance']
        var_bounds = npz['variance_normalisation_bounds']
        input_bounds = npz['input_bounds']
        target_mean = npz['target_mean']
        target_std = npz['target_std']

    # Make the GP
    gp = gpflow.models.SGPR(
        data=(X_train, y_train),
        inducing_variable=Z,
        kernel=gpflow.kernels.SquaredExponential(
            lengthscales=kernel_lengthscales,
            variance=kernel_variance
        )
    )
    gp.likelihood.variance.assign(likelihood_variance)

    if error_aware_acqf:
        with np.load(data_err_file, allow_pickle=True) as npz:
            X_train_err = npz['X_train']
            X_test_err = npz['X_test']
            err_train = npz['y_train']
            err_test = npz['y_test']

        with np.load(gp_err_file, allow_pickle=True) as npz:
            err_Z = npz['Z']
            err_kernel_lengthscales = npz['kernel_lengthscales']
            err_kernel_variance = npz['kernel_variance']
            err_likelihood_variance = npz['likelihood_variance']
            err_var_bounds = npz['variance_normalisation_bounds']
            err_input_bounds = npz['input_bounds']
            err_target_mean = npz['target_mean']
            err_target_std = npz['target_std']

            # Make the GP
            gp_err = gpflow.models.SGPR(
                data=(X_train_err, err_train),
                inducing_variable=err_Z,
                kernel=gpflow.kernels.SquaredExponential(
                    lengthscales=err_kernel_lengthscales,
                    variance=err_kernel_variance
                )
            )
            gp_err.likelihood.variance.assign(err_likelihood_variance)

    """ 
    Choose a value for fmin.
    In pratice, it seems that for a very small value, the EI gradients
    are very small, so the optimization doesn't converge.
    Choosing a low-ish percentile seems to be a good compromise.
    """
    fmin = np.percentile(y_train, 10)
    LOGGER.info(f"Using fmin={fmin:.2f}")

    # Choose other bounds/cutoffs
    good_point_cutoff = np.percentile(y_train, 20)
    LOGGER.info(f"Using good point cutoff={good_point_cutoff:.2f}")
    data_bounds = np.percentile(np.abs(X_train), 99.9)  # To account for outliers
    LOGGER.info(f"Data bound of {data_bounds} found")
    data_bounds *= 1.1
    LOGGER.info(f"Using data bound of {data_bounds}")

    # Run the optimization, with a mix of random and good points
    LOGGER.info("\n### Starting optimization ### \n")

    if error_aware_acqf:
        func = functools.partial(
            neg_eaei_and_grad, gp_f=gp, gp_r=gp_err, fmin=fmin,
            input_bounds=input_bounds, err_input_bounds=err_input_bounds,
            target_mean=target_mean, target_std=target_std,
            err_target_mean=err_target_mean, err_target_std=err_target_std,
            var_bounds=var_bounds, err_var_bounds=err_var_bounds)
    else:
        func = functools.partial(neg_ei_and_grad, gp=gp, fmin=fmin)

    latent_pred, ei_vals = robust_multi_restart_optimizer(
        func,
        X_train, y_train,
        num_pts_to_return=n_out,
        num_random_starts=n_starts // 2,
        num_good_starts=n_starts - n_starts // 2,
        good_point_cutoff=good_point_cutoff,
        bounds_abs=data_bounds,
        logger=LOGGER,
        early_stop=early_stopping,
        use_tqdm=True
    )
    LOGGER.info(f"Done optimization! {len(latent_pred)} results found\n\n.")

    # Save results
    latent_pred = np.array(latent_pred, dtype=np.float32)
    np.save(save_file, latent_pred)

    # Make some gp predictions in the log file
    LOGGER.info("EI results:")
    LOGGER.info(ei_vals)

    mu, var = gp.predict_f(latent_pred)
    LOGGER.info("mu at points:")
    LOGGER.info(list(mu.numpy().ravel()))
    LOGGER.info("var at points:")
    LOGGER.info(list(var.numpy().ravel()))
    if error_aware_acqf:
        mu_r, var_r = gp_err.predict_f(latent_pred)
        LOGGER.info("mu_err at points:")
        LOGGER.info(list(mu_r.numpy().ravel()))
        LOGGER.info("var_err at points:")
        LOGGER.info(list(var_r.numpy().ravel()))

    LOGGER.info("\n\nEND OF SCRIPT!")

    return latent_pred


if __name__ == "__main__":
    args = parser.parse_args()
    np.random.seed(args.seed)
    gp_opt(
        gp_file=args.gp_file,
        data_file=args.data_file,
        save_file=args.save_file,
        n_out=args.n_out,
        logfile=args.logfile,
        n_starts=args.n_starts,
        early_stopping=args.early_stopping,
        gp_err_file=args.gp_err_file,
        data_err_file=args.data_err_file,
    )

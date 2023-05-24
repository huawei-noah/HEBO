from pickle import NONE
import shutil
import logging
import mlflow
import platform

import numpy as np

import init
import datasets
import algorithms
from common import Config
from acquisition_optimizer import MPAcquisitionOptimizer
from mlflow_logging import MlflowLogger
import socket
from pid import PidFile
import os
import sys
import tempfile

def main(mlflow_logger):

    # Machine Related logging
    logging.info('Platform: %s', platform.uname())
    logging.info('Processor: %s', platform.processor())
    logging.info('Python: %s/%s', platform.python_version(), platform.python_implementation())

    # Log LAPACK Information
    #logging.info("Blas Library: %s", np.__config__.get_info('blas_opt_info')['libraries'])
    #logging.info("Lapack Library: %s", np.__config__.get_info('lapack_opt_info')['libraries'])
    
    # Temp files
    logging.info("temp_dir: {}".format(Config().base_path))
    logging.info("Host Name: {}".format(platform.node()))
    
    args = init.args_parse()

    logging.info(f"{args}")
    
    # Log all parameters to mlflow
    printable_params = frozenset(["algorithm_random_seed", "data_random_seed"])
    for param_key in args:
        if param_key in printable_params:
            mlflow_logger.log_param(param_key, args[param_key])

    # We change the name of the experiment
    tags = {
        "mlflow.runName"    : "{}".format(args["algorithm"]),
        "host_name"         : "{}".format(socket.gethostname().split(".")[0]),
        "hash_exe"          : args["hash_exe"],
        "hash_data"         : args["hash_data"]
    }

    # Setup the data
    # =========================================================
    # Get the loader of the dataset type
    MetaDataTypeLoader = datasets.MetaLoader.get_loader_constructor(args["which_data"])
    # Now load the actual Loader
    Loader = MetaDataTypeLoader.get_loader_constructor(**args)
    dataLoader = Loader(**args)
    fn, soln = dataLoader.load()
    known_optimum = False
    if soln == None:
        logging.info("Computing f_min")
        optimizer = MPAcquisitionOptimizer(fn.domain, fn, [], None, max_eval=-1)
        cfn = fn.make_component_function()
        soln = optimizer.optimize(cfn)
        x_best, f_min, cost = soln
        assert(np.isclose(cfn(x_best), f_min))
        assert(np.isclose(fn.eval(x_best), f_min))
        dataLoader.save(fn, soln)
    else:
        known_optimum = True
        x_best, f_min, cost = soln

    logging.info("f_min = {}".format(f_min))
    logging.info("x_best = {}".format(x_best))
    logging.info("cost = {}".format(cost))
    
    #TODO Sanity checks
    tags["f_min"] = f_min
    mlflow_logger.set_tags(tags)
    
    mlflow_logger.update_truth(f_min, fn.graph)
    fn.mlflow_logging = mlflow_logger

    # Run the algorithm
    # =========================================================
    
    # We random the lengthscale and report it
    MetaAlgoTypeLoader = algorithms.MetaLoader.get_loader_constructor(args["which_algorithm"])
    Algorithm = MetaAlgoTypeLoader.get_constructor(args["algorithm"])
    algorithm = Algorithm(fn=fn, **args)
    if known_optimum:
        algorithm.run(stopping_Y=f_min)
    else:
        algorithm.run()

    # If best regret was reached, no need to run further iterations
    if mlflow_logger.y_best == f_min:
        for _ in range(args["n_iter"] - mlflow_logger.t_y):
            mlflow_logger.log_y(mlflow_logger.y_best)

    # check if the N is in order.
    if mlflow_logger.t_y != args["n_iter"]:
        raise Exception("Missing some iterations.")

def cleanup(mlflow_logger):
    # Clean up, log artifacts and remove directory
    mlflow_logger.log_artifacts(Config().base_path)
    shutil.rmtree(Config().base_path)

if __name__ == '__main__':

    init.logger()
    mlflow_logger = MlflowLogger()
    tmp_code_dir_path = os.path.dirname(os.path.realpath(__file__))
    logging.info(f"Path location - {tmp_code_dir_path}, tmp location - {tempfile.gettempdir()}")
    # Ensure that the code cannot fail and log everything...
    try:

        # Check if running in daemon mode 
        if len(sys.argv) == 3:
            hash_exe=os.path.splitext(os.path.basename(sys.argv[1]))[0]
            with PidFile(hash_exe, sys.argv[2]) as p:
                main(mlflow_logger)
        else:
            main(mlflow_logger)
            
    except Exception as e:
        mlflow_logger.set_tag("Error", str(type(e).__name__))
        logging.exception("Exception")
        cleanup(mlflow_logger)
        
        # Exit with Error code.
        sys.exit(e)

    cleanup(mlflow_logger)

#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
# =============

import logging
import os
import numpy as np
from common import Config

logging_types = ['local', 'server']
# Determines if we are logging on the server or not
def logger(env_key='LOGGING_TYPE'):
    
    logging_type = os.getenv(env_key, 'local')

    log_formatter = logging.Formatter('%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    info_file_handler = logging.FileHandler(Config().log_file("info.log"))
    info_file_handler.setFormatter(log_formatter)
    info_file_handler.setLevel(logging.INFO)
    root_logger.addHandler(info_file_handler)
    
    error_file_handler = logging.FileHandler(Config().log_file("error.log"))
    error_file_handler.setFormatter(log_formatter)
    error_file_handler.setLevel(logging.ERROR)
    root_logger.addHandler(error_file_handler)
    
    if logging_type == 'local':
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(log_formatter)
        consoleHandler.setLevel(logging.INFO)
        root_logger.addHandler(consoleHandler)
        
# =====================================================================
import datasets
import algorithms
import yaml
import sys
import hashlib
import json
import argparse

# Argument Constants
syn_graphtype_list = datasets.Synthetic.loader_ids
nas_bench_list = datasets.NAS.loader_ids
# Check avaliable fcnet benchmark files
fcnet_benchmark_list = Config().list_fcnet()

# Must return a 
def args_parse():

    args_dict = yaml.load(open(sys.argv[1]), yaml.FullLoader)
    args_dict["hash_exe"] = hashlib.sha1(json.dumps(args_dict, sort_keys=True).encode()).hexdigest()

    print(args_dict)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=None)
    parser.add_argument("--sub_benchmark", default=None)

    args, _ = parser.parse_known_args()
    if args.seed is not None:
        args_dict['algorithm_type']['Algorithm']['algorithm_random_seed'] = int(args.seed)

    which_datas = list(args_dict["data_type"].keys())
    if len(which_datas) != 1:
        logging.error('There should be only one data type declared.')

    which_algorithms = list(args_dict["algorithm_type"].keys())
    if len(which_algorithms) != 1:
        logging.error('There should be only one algorithm type declared.')

    if args.sub_benchmark is not None:
        sub_benchmark_type, sub_benchmark_name = args.sub_benchmark.split(":")
        args_dict["data_type"][which_datas[0]][sub_benchmark_type] = sub_benchmark_name

    # Process the 2 types, data and algo type
    args_dict["which_data"] = which_datas[0]
    args_dict["which_algorithm"] = which_algorithms[0]

    args_dict["hash_data"] = hashlib.sha1(json.dumps(args_dict["data_type"], sort_keys=True).encode()).hexdigest()

    # Flatten to the parameters
    args_dict.update(args_dict["data_type"][args_dict["which_data"]])
    del args_dict["data_type"]
    args_dict.update(args_dict["algorithm_type"][args_dict["which_algorithm"]])
    del args_dict["algorithm_type"]

    # turn it into lambda if string
    if type(args_dict["exploration_weight"]) == str:
        args_dict["exploration_weight"] =  eval(args_dict["exploration_weight"])

    return args_dict

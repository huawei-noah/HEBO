#!/usr/bin/env bash
PYTHONPATH=./ python ./tests/compare_robust_optimization.py --input_distribution beta \
--function_name CustomK --n_iter 50 --use_multiprocess --use_gpu
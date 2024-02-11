#!/usr/bin/env bash
PYTHONPATH=./ python ./tests/compare_optimization_in_push_world.py --n_iter 50 --use_multiprocess \
--use_gpu --debug_mode
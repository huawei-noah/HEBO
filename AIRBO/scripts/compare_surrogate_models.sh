#!/usr/bin/env bash
PYTHONPATH=./ python ./tests/compare_surrogate_models.py --input_distribution step_chi2 --function_name RKHS --use_gpu
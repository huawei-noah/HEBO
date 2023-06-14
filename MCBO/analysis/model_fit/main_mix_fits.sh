#!/bin/bash

surrogates="gp_to gp_o gp_hed"
data_method_source=ga

for task in xgboost_opt aig_optimization_hyp svm_opt "ackley-53"; do
  for seed in {42..56}; do
    i=0
    for n_observe in 150; do
      absolut_id=$((i + 1))
      absolut_dir=/home/antoine/absoluts/Absolut${absolut_id}/src/AbsolutNoLib
      device=$((i % 4))
      taskset_start=$((i * 3))
      taskset_end=$((taskset_start + 2))
      cmd="python ./analysis/model_fit/main_model_fit.py --task_ids $task --data_method_source $data_method_source --surrogate_ids $surrogates --device $device --absolut_dir $absolut_dir --n_observes $n_observe --seeds $seed"
      taskset -c $taskset_start-$taskset_end $cmd &
      i=$((i + 1))
    done
    wait
  done
done

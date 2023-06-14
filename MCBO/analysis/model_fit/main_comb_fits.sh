#!/bin/bash

surrogates="gp_to gp_o gp_hed gp_ssk gp_diff"
data_method_source=sa

for task in ackley mig_optimization aig_optimization pest antibody_design rna_inverse_fold; do
  for seed in {42..51}; do
    i=0
    for n_observe in 150; do
      absolut_id=$((i + 1))
      absolut_dir=/home/antoine/absoluts/Absolut${absolut_id}/src/AbsolutNoLib
      device=$((i % 4))
      taskset_start=$((i * 3))
      taskset_end=$((taskset_start + 2))
      cmd="python ./analysis/model_fit/test_fits.py --task_names $task --data_method_source $data_method_source --surrogate_ids $surrogates --device $device --absolut_dir $absolut_dir --n_observes $n_observe --seeds $seed"
      taskset -c $taskset_start-$taskset_end $cmd &
      i=$((i + 1))
    done
    wait
  done
done

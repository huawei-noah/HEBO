#!/bin/bash

mapping=fpga

ref_abc_seq=resyn2

seq_length=6

n_trials=15
random_sampling_id='latin-hypercube'
seed=0

overwrite=1
if (( overwrite == 0 )); then overwrite=''; else overwrite='--overwrite'; fi

designs_group_id=aux_test_abc_graph
n_parallel=10

for action_space_id in standard; do
  cmd="python ./core/algos/random/main_multi_random.py \
              --designs_group_id $designs_group_id \
              --n_parallel $n_parallel \
              --seq_length $seq_length \
              --mapping $mapping \
              --action_space_id $action_space_id \
              --ref_abc_seq $ref_abc_seq \
              --n_trials $n_trials \
              --random_sampling_id $random_sampling_id \
              --seed $seed $overwrite"
  echo $cmd
  $cmd
done

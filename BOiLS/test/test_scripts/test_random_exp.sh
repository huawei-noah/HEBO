#!/bin/bash

path_design_file='./data/epfl-benchmark/arithmetic/'
mapping=fpga

ref_abc_seq=resyn2

seq_length=5

n_trials=10
random_sampling_id='latin-hypercube'
seed=0

overwrite=0
if (( overwrite == 0 )); then overwrite=''; else overwrite='--overwrite'; fi

N=4
for i in 1 2 3; do
  frac_part="$i/4"
  for action_space_id in standard; do
    designs_group_id=aux_test_designs_group
    cmd="python ./core/algos/random/main_random.py \
                --designs_group_id $designs_group_id \
                --frac_part $frac_part \
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
done
#!/bin/bash

mapping=fpga
lut_inputs=6

ref_abc_seq=resyn2

n_trials=1000
seq_length=20
random_sampling_id='latin-hypercube'
overwrite=0
if ((overwrite == 0)); then overwrite=''; else overwrite='--overwrite'; fi
use_yosys=1

n_parallel=8

for action_space_id in extended; do
  for designs_group_id in max sin sqrt adder log2 multiplier square bar div hyp; do
    for seed in {0..4}; do
        cmd="python ./core/algos/random/main_multi_random.py \
              --designs_group_id $designs_group_id \
              --n_parallel $n_parallel \
              --seq_length $seq_length \
              --mapping $mapping \
              --lut_inputs $lut_inputs \
              --use_yosys $use_yosys \
              --action_space_id $action_space_id \
              --ref_abc_seq $ref_abc_seq \
              --n_trials $n_trials \
              --random_sampling_id $random_sampling_id \
              --seed $seed $overwrite"
        echo $cmd

        $cmd &
      done
    wait
  done
done

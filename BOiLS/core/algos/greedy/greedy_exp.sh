#!/bin/bash

# --- EDA setup ---

mapping=fpga       # parameter of the EDA mapping
lut_inputs=6       # parameter of the EDA mapping
use_yosys=1        # whether to use yosys-abc command or the abc_py package for the evaluation

# --- Exp setup ---
# Choose the logic synthesis sequence baseline
ref_abc_seq=resyn2 # the most-used one
#ref_abc_seq=init   # simply the empty sequence -> compare against doing nothing

seq_length=20 # the length of the sequence of operations (also the dimensionality of the problem)

# --- Greedy setup ---

# --- Run parameters ---

overwrite=0
if ((overwrite == 0)); then overwrite=''; else overwrite='--overwrite'; fi

lut_inputs=6
n_parallel=4

circuits=(adder hyp square sqrt multiplier div log2 max bar sin)

for action_space_id in extended standard; do # can run for several action spaces (available operations in sequences)
  for i in $(seq 1 ${#circuits[@]}); do
    #      taskset_start=$((i * 8))
    #      taskset_end=$((taskset_start + 7))
    designs_group_id=${circuits[i-1]}

    for seed in {0..4}; do # can run for several seeds
      #      for objective in both lut; do
      for objective in both; do
        cmd="python ./core/algos/greedy/main_greedy_exp.py \
              --designs_group_id $designs_group_id \
              --n_parallel $n_parallel \
              --seq_length $seq_length \
              --use_yosys $use_yosys \
              --objective $objective \
              --mapping $mapping --lut_inputs $lut_inputs \
              --action_space_id $action_space_id \
              --ref_abc_seq $ref_abc_seq \
              --seed $seed $overwrite"
        echo $cmd
        $cmd &
      done
    done
    wait
  done
done

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

n_parallel=15 # Number of sequences to evaluate in parallel (since we don't do batch BO, no need to set > 1)

# --- HEBO setup ---

n_total_evals=1000 # Number of acquisitions in total (number of sequences that we will evaluate)
n_initial=20      # Number of initial random sequences to evaluate to build the first surrogate model

# --- Run parameters ---

overwrite=0
if ((overwrite == 0)); then overwrite=''; else overwrite='--overwrite'; fi

for action_space_id in extended; do # can run for several action spaces (available operations in sequences)
#  for designs_group_id in adder sin log2 multiplier square sqrt max bar div hyp; do  # 48 - 0
  for designs_group_id in hyp; do  # 48 - 2
    i=0
    for seed in {0..4}; do # can run for several seeds
#    for seed in {0..0}; do # can run for several seeds
#      for objective in level; do
      for objective in both; do
        cmd="python ./core/algos/bo/hebo/main_multi_hebo.py \
                    --designs_group_id $designs_group_id \
                    --n_parallel $n_parallel \
                    --seq_length $seq_length \
                    --mapping $mapping \
                    --action_space_id $action_space_id \
                    --ref_abc_seq $ref_abc_seq \
                    --n_total_evals $n_total_evals \
                    --n_initial $n_initial \
                    --lut_inputs $lut_inputs \
                    --use_yosys $use_yosys \
                    --objective $objective \
                    --seed $seed $overwrite"
        taskset_start=$((i * 15))
        taskset_end=$((taskset_start + 14))
        echo "$taskset_start-$taskset_end | $designs_group_id $objective $seed"
        taskset -c $taskset_start-$taskset_end $cmd &
        echo $cmd
        i=$((i + 1))
      done
    done
    wait
  done
done

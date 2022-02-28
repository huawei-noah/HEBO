#!/bin/bash

# --- EDA setup ---

mapping=fpga       # parameter of the EDA mapping
lut_inputs=6      # parameter of the EDA mapping
use_yosys=1        # whether to use yosys-abc command or the abc_py package for the evaluation

# --- Exp setup ---
# Choose the logic synthesis sequence baseline
ref_abc_seq=resyn2 # the most-used one
#ref_abc_seq=init   # simply the empty sequence -> compare against doing nothing

seq_length=20 # the length of the sequence of operations (also the dimensionality of the problem)

large_n_evals=20000
# --- GA setup ---
if ((seq_length <= 10)); then n_total_evals=100; else n_total_evals=$large_n_evals; fi
#pop_size=20
pop_size=75
parents_portion=0.1
elit_ration=0.0003
mutation_probability=0.25
crossover_probability=.5
crossover_type=two_point

# --- Run parameters ---

overwrite=0
if ((overwrite == 0)); then overwrite=''; else overwrite='--overwrite'; fi

circuits=(adder div sqrt)
n_parallels=(3 60 15)

taskset_end=-1

for action_space_id in extended; do # can run for several action spaces (available operations in sequences)
  for objective in lut; do
    for i in $(seq 1 ${#circuits[@]}); do
      n_parallel=${n_parallels[i - 1]}
      taskset_start=$((taskset_end + 1))
      taskset_end=$((taskset_start + n_parallel - 1))
      designs_group_id=${circuits[i - 1]}

      for seed in {0..0}; do # can run for several seeds
        cmd="taskset -c $taskset_start-$taskset_end python ./core/algos/genetic/sga/main_multi_sga.py \
              --designs_group_id $designs_group_id \
              --n_parallel $n_parallel \
              --seq_length $seq_length \
              --use_yosys $use_yosys \
              --objective $objective \
              --mapping $mapping --lut_inputs $lut_inputs \
              --action_space_id $action_space_id \
              --ref_abc_seq $ref_abc_seq \
              --n_total_evals $n_total_evals \
              --pop_size $pop_size \
               --parents_portion $parents_portion \
              --elit_ration $elit_ration \
              --crossover_probability $crossover_probability \
              --crossover_type $crossover_type \
              --mutation_probability $mutation_probability \
              --seed $seed $overwrite"
        echo $cmd
        taskset -c $taskset_start-$taskset_end $cmd &
      done
    done
  done
  wait
done

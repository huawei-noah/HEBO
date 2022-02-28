#!/bin/bash

# --- EDA setup ---

mapping=fpga       # parameter of the EDA mapping
lut_inputs=4       # parameter of the EDA mapping
use_yosys=1        # whether to use yosys-abc command or the abc_py package for the evaluation

# --- Exp setup ---
# Choose the logic synthesis sequence baseline
ref_abc_seq=resyn2 # the most-used one
#ref_abc_seq=init   # simply the empty sequence -> compare against doing nothing

seq_length=20 # the length of the sequence of operations (also the dimensionality of the problem)

# --- NSGA2 setup ---
if ((seq_length <= 10)); then n_total_evals=100; else n_total_evals=$large_n_evals; fi
pop_size=80
n_gen=1000

# --- Run parameters ---

overwrite=0
if ((overwrite == 0)); then overwrite=''; else overwrite='--overwrite'; fi

circuits=(hyp div log2 sqrt multiplier adder square max bar sin) # .46 - 0
circuits=(div)                                                   # .46 - 1
circuits=(log2)                                                  # .46 - 2
circuits=(sqrt multiplier adder square max bar sin)              # .46 - 3
circuits=(adder max bar sin)                                     # .46 - 3
circuits=(log2 div)                                              # .46 - 3
circuits=(log2 div hyp)
n_parallels=(2 4 10)

taskset_end=-1

for action_space_id in extended; do # can run for several action spaces (available operations in sequences)
  for i in $(seq 1 ${#circuits[@]}); do
    n_parallel=${n_parallels[i - 1]}
    taskset_start=$((taskset_end + 1))
    taskset_end=$((taskset_start + n_parallel - 1))
    designs_group_id=${circuits[i - 1]}

    for seed in {0..0}; do # can run for several seeds
      cmd="python ./core/algos/genetic/nsga2/main_multi_nsga2.py\
              --designs_group_id $designs_group_id \
              --n_parallel $n_parallel \
              --seq_length $seq_length \
              --use_yosys $use_yosys \
              --mapping $mapping --lut_inputs $lut_inputs \
              --action_space_id $action_space_id \
              --ref_abc_seq $ref_abc_seq \
              --pop_size $pop_size \
              --n_gen $n_gen \
              --seed $seed $overwrite"
      echo "taskset -c $taskset_start-$taskset_end $cmd"
      taskset -c $taskset_start-$taskset_end $cmd &
    done
  done
  wait
done

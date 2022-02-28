#!/bin/bash

# --- EDA setup ---

mapping=fpga # parameter of the EDA mapping
lut_inputs=6 # parameter of the EDA mapping
use_yosys=1  # whether to use yosys-abc command or the abc_py package for the evaluation

# --- Exp setup ---
ref_abc_seq=resyn2
objective='both'

#action_space_ind=0
#action_space_ids=(extended standard)
#action_space_id=${action_space_ids[action_space_ind]}
seq_length=20

# --- GRiLLS setup ---
n_episodes=1000

overwrite=0
if ((overwrite == 0)); then overwrite=''; else overwrite='--overwrite'; fi

i=0
for designs_group_id in sin; do
  for action_space_id in extended; do
    for ((seed = 4; seed < 5; seed++)); do
#      taskset_start=$((i * 10))
      taskset_start=15
#      taskset_end=$((taskset_start + 9))
      taskset_end=20
      i=$((i + 1))
      echo "$taskset_start-$taskset_end | $designs_group_id $objective $seed"

      cmd="python ./core/algos/GRiLLS/main_multi_grills_exp.py \
                --designs_group_id $designs_group_id \
                --seq_length $seq_length \
                --mapping $mapping \
                --lut_inputs $lut_inputs \
                --use_yosys $use_yosys \
                --objective $objective
                --action_space_id $action_space_id \
                --ref_abc_seq $ref_abc_seq \
                --n_episodes $n_episodes \
                --seed $seed $overwrite"
      echo $cmd
      $cmd
    done
  done
done
wait

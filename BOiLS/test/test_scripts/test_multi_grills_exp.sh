#!/bin/bash

# --- EDA setup ---

mapping=fpga       # parameter of the EDA mapping
lut_inputs=6       # parameter of the EDA mapping
use_yosys=1        # whether to use yosys-abc command or the abc_py package for the evaluation

# --- Exp setup ---
ref_abc_seq=resyn2
objective='both'

#action_space_ind=0
#action_space_ids=(extended standard)
#action_space_id=${action_space_ids[action_space_ind]}
seq_length=20

# --- GRiLLS setup ---
n_episodes=20

seed=0

overwrite=1
if (( overwrite == 0 )); then overwrite=''; else overwrite='--overwrite'; fi

for designs_group_id in max; do #  adder div sqrt  #    adder  log2 multiplier #
  for action_space_id in extended; do
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

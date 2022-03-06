#!/bin/bash

# --- EDA setup ---

mapping=fpga       # parameter of the EDA mapping
lut_inputs=6       # parameter of the EDA mapping
use_yosys=1        # whether to use yosys-abc command or the abc_py package for the evaluation

ref_abc_seq=resyn2
#ref_abc_seq=init


#seq_to_test='[6,0,2,6,0,1,6,3,1,6]'
#seq_origin=resyn2


seq_length=60
seq_to_test='[6,0,2,6,0,1,6,3,1,6,6,0,2,6,0,1,6,3,1,6,6,0,2,6,0,1,6,3,1,6,6,0,2,6,0,1,6,3,1,6,6,0,2,6,0,1,6,3,1,6,6,0,2,6,0,1,6,3,1,6]'
seq_origin=resyn2_6

overwrite=0
if (( overwrite == 0 )); then overwrite=''; else overwrite='--overwrite'; fi

n_parallel=10

action_space_id=extended

for designs_group_id in hyp adder max sin square sqrt multiplier div log2 bar; do

  cmd="python ./core/algos/seqs_test/main_multi_seqs_test.py \
              --designs_group_id $designs_group_id \
              --n_parallel $n_parallel \
              --seq_length $seq_length \
              --mapping $mapping \
              --lut_inputs $lut_inputs \
              --action_space_id $action_space_id \
              --ref_abc_seq $ref_abc_seq \
              --seq_to_test $seq_to_test \
              --use_yosys $use_yosys \
              --seq_origins $seq_origin $overwrite"
  echo $cmd
  $cmd &
done
wait
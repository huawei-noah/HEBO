#!/bin/bash

path_design_file='./data/epfl-benchmark/arithmetic/'
mapping=fpga

ref_abc_seq=resyn2

#action_space_ind=0
#action_space_ids=(extended standard)
#action_space_id=${action_space_ids[action_space_ind]}
seq_length=8

n_total_evals=50
pop_size=15
seed=0

overwrite=1
if (( overwrite == 0 )); then overwrite=''; else overwrite='--overwrite'; fi

for design in adder; do #  bar div sqrt  #    adder  log2 multiplier #
  for action_space_id in standard extended; do
    design_file=$path_design_file$design.v
    cmd="python ./core/main_sga.py \
                --design_file $design_file \
                --seq_length $seq_length \
                --mapping $mapping \
                --action_space_id $action_space_id \
                --ref_abc_seq $ref_abc_seq \
                --n_total_evals $n_total_evals \
                --pop_size $pop_size \
                --seed $seed $overwrite"
    echo $cmd
    $cmd
  done
done

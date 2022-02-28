#!/bin/bash

path_design_file='./data/epfl-benchmark/arithmetic/'
mapping=fpga

ref_abc_seq=resyn2

#action_space_ind=0
#action_space_ids=(extended standard)
#action_space_id=${action_space_ids[action_space_ind]}
seq_length=10

n_episodes=10
seed=0

overwrite=1
if (( overwrite == 0 )); then overwrite=''; else overwrite='--overwrite'; fi

design_files="['adder', 'bar']"

for design in adder; do #  adder div sqrt  #    adder  log2 multiplier #
  for action_space_id in extended; do
    design_file=$path_design_file$design.v
    cmd="python ./core/main_grills.py \
                --design_file $design_file \
                --seq_length $seq_length \
                --mapping $mapping \
                --action_space_id $action_space_id \
                --ref_abc_seq $ref_abc_seq \
                --n_episodes $n_episodes \
                --seed $seed $overwrite"
    echo $cmd
    $cmd
  done
done

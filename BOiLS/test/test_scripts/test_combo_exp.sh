#!/bin/bash

path_design_file='./data/epfl-benchmark/arithmetic/'
mapping=fpga

ref_abc_seq=resyn2

#action_space_ind=0
#action_space_ids=(extended standard)
#action_space_id=${action_space_ids[action_space_ind]}
seq_length=10

n_eval=10
n_initial=5
seed=0

overwrite=0
if (( overwrite == 0 )); then overwrite=''; else overwrite='--overwrite'; fi

design_files="['adder','bar']"

for design in sin; do #  adder div sqrt  #    adder  log2 multiplier #
  for action_space_id in extended; do
    design_file=$path_design_file$design.v
    cmd="python ./core/algos/bo/main_combo.py \
                --design_file $design_file \
                --design_files $design_files \
                --seq_length $seq_length \
                --mapping $mapping \
                --action_space_id $action_space_id \
                --ref_abc_seq $ref_abc_seq \
                --n_eval $n_eval \
                --n_initial $n_initial \
                --seed $seed $overwrite"
    $cmd
    echo $cmd
  done
done

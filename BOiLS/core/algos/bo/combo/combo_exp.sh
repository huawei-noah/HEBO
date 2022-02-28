#!/bin/bash

path_design_file='./data/epfl-benchmark/arithmetic/'
mapping=fpga

ref_abc_seq=resyn2

#action_space_ind=0
#action_space_ids=(extended standard)
#action_space_id=${action_space_ids[action_space_ind]}
seq_length=10

n_eval=200
n_initial=20
seed=0

overwrite=0
if ((overwrite == 0)); then overwrite=''; else overwrite='--overwrite'; fi

design_files="['${path_design_file}adder.v','${path_design_file}sin.v','${path_design_file}sqrt.v','${path_design_file}log2.v','${path_design_file}hyp.v']"
design_files="['${path_design_file}max.v','${path_design_file}bar.v','${path_design_file}square.v','${path_design_file}multiplier.v','${path_design_file}div.v']"

design=adder
design=max

for action_space_id in standard; do
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
  echo $cmd
  $cmd
done

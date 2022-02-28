#!/bin/bash

path_design_file='./data/epfl-benchmark/arithmetic/'
mapping=fpga

ref_abc_seq=resyn2

#action_space_ind=0
#action_space_ids=(extended standard standard)
#action_space_id=${action_space_ids[action_space_ind]}
seq_length=10

pop_size=40
n_gen=25
seed=0

overwrite=1
if (( overwrite == 0 )); then overwrite=''; else overwrite='--overwrite'; fi


for design in hyp bar div sqrt; do #    adder  log2 multiplier # sin max square hyp #
  for action_space_id in extended; do
    design_file=$path_design_file$design.v
    cmd="python ./core/main_nsga2.py \
                --design_file $design_file \
                --seq_length $seq_length \
                --mapping $mapping \
                --action_space_id $action_space_id \
                --ref_abc_seq $ref_abc_seq \
                --pop_size $pop_size \
                --n_gen $n_gen \
                --seed $seed $overwrite"
    echo $cmd
    $cmd &
  done
done

sleep 2h

for design in adder log2 multiplier; do #bar div sqrt # sin max square hyp #
  for action_space_id in standard; do
    design_file=$path_design_file$design.v
    cmd="python ./core/main_nsga2.py \
                --design_file $design_file \
                --seq_length $seq_length \
                --mapping $mapping \
                --action_space_id $action_space_id \
                --ref_abc_seq $ref_abc_seq \
                --pop_size $pop_size \
                --n_gen $n_gen \
                --seed $seed $overwrite"
    echo $cmd
    $cmd &
  done
done

sleep 2h

#  hyp
for design in sin max square; do #  bar div sqrt    adder  log2 multiplier #
  for action_space_id in standard; do
    design_file=$path_design_file$design.v
    cmd="python ./core/main_nsga2.py \
                --design_file $design_file \
                --seq_length $seq_length \
                --mapping $mapping \
                --action_space_id $action_space_id \
                --ref_abc_seq $ref_abc_seq \
                --pop_size $pop_size \
                --n_gen $n_gen \
                --seed $seed $overwrite"
    echo $cmd
    $cmd &
  done
done

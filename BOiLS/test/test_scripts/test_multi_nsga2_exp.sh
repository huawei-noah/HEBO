#!/bin/bash

mapping=fpga

ref_abc_seq=resyn2

seq_length=10

n_gen=3
pop_size=4
seed=0

eta_cross=2
eta_mute=5
prob_cross=.8
selection='random'

overwrite=0
if (( overwrite == 0 )); then overwrite=''; else overwrite='--overwrite'; fi

designs_group_id=aux_test_designs_group
n_parallel=10

for action_space_id in standard; do
  cmd="python ./core/algos/genetic/nsga2/main_multi_nsga2.py \
              --designs_group_id $designs_group_id \
              --n_parallel $n_parallel \
              --seq_length $seq_length \
              --mapping $mapping \
              --action_space_id $action_space_id \
              --ref_abc_seq $ref_abc_seq \
              --n_gen $n_gen \
              --pop_size $pop_size \
              --eta_cross $eta_cross \
              --eta_mute $eta_mute \
              --prob_cross $prob_cross \
              --selection $selection \
              --seed $seed $overwrite"
  echo $cmd
  $cmd
done

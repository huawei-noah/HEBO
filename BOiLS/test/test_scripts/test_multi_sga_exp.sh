#!/bin/bash

mapping=fpga

ref_abc_seq=resyn2

seq_length=10

n_total_evals=25
pop_size=4
seed=0

overwrite=0
if (( overwrite == 0 )); then overwrite=''; else overwrite='--overwrite'; fi

design_files_id=aux_test_designs_group
n_parallel=10

for action_space_id in standard; do
  cmd="python ./core/algos/genetic/main_multi_sga.py \
              --design_files_id $design_files_id \
              --n_parallel $n_parallel \
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

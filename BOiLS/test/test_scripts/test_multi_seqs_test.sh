#!/bin/bash

mapping=fpga

ref_abc_seq=resyn2

seq_length=10

seq_to_test='[7,10,3,6,8,3,10,4,10,5]'
seq_origin=test

overwrite=1
if (( overwrite == 0 )); then overwrite=''; else overwrite='--overwrite'; fi

designs_group_id=aux_test_designs_group
n_parallel=2

for action_space_id in extended; do
  cmd="python ./core/algos/seqs_test/main_multi_seqs_test.py \
              --designs_group_id $designs_group_id \
              --n_parallel $n_parallel \
              --seq_length $seq_length \
              --mapping $mapping \
              --action_space_id $action_space_id \
              --ref_abc_seq $ref_abc_seq \
              --seq_to_test $seq_to_test \
              --seq_origins $seq_origin $overwrite"
  echo $cmd
  $cmd
done

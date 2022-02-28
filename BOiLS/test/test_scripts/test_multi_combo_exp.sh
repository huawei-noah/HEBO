#!/bin/bash

mapping=fpga

ref_abc_seq=resyn2

#action_space_ind=0
#action_space_ids=(extended standard)
#action_space_id=${action_space_ids[action_space_ind]}
seq_length=10

n_eval=12
n_initial=10
seed=0

n_parallel=10
designs_group_id=aux_test_designs_group

overwrite=0
if (( overwrite == 0 )); then overwrite=''; else overwrite='--overwrite'; fi
device=0

for action_space_id in extended; do
  cmd="python ./core/algos/bo/main_multi_combo.py \
              --designs_group_id $designs_group_id \
              --n_parallel $n_parallel \
              --seq_length $seq_length \
              --mapping $mapping \
              --action_space_id $action_space_id \
              --ref_abc_seq $ref_abc_seq \
              --n_eval $n_eval \
              --n_initial $n_initial \
              --device $device \
              --seed $seed $overwrite"
  $cmd
  echo $cmd
done

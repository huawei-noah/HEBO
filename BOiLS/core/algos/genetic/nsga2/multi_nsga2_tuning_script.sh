#!/bin/bash

mapping=fpga

ref_abc_seq=resyn2

seq_length=10

n_gen=10
pop_size=75
seed=0

n_acq=30
search_space_id=multi_nsga2_space_1

designs_group_id=hisi174_train
n_parallel=20

overwrite=0
if ((overwrite == 0)); then overwrite=''; else overwrite='--overwrite'; fi

action_space_id=extended

cmd="python ./core/algos/genetic/nsga2/multi_nsga2_tuning.py \
      --search_space_id $search_space_id \
      --n_acq $n_acq \
      --designs_group_id $designs_group_id \
      --n_parallel $n_parallel \
      --seq_length $seq_length \
      --mapping $mapping \
      --action_space_id $action_space_id \
      --ref_abc_seq $ref_abc_seq \
      --n_gen $n_gen \
      --pop_size $pop_size \
      --seed $seed $overwrite"
echo $cmd
$cmd

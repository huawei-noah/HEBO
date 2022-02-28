#!/bin/bash

mapping=fpga

ref_abc_seq=resyn2

seq_length=10

n_trials=3
data_n_trials=1000

random_sampling_id='latin-hypercube'
seed=0
data_seed=0

log_transf=1
if (( log_transf == 0 )); then log_transf=''; else log_transf='--log_transf'; fi

sample=0
if (( sample == 0 )); then sample=''; else sample='--sample'; fi

overwrite=0
if (( overwrite == 0 )); then overwrite=''; else overwrite='--overwrite'; fi


designs_group_id=aux_test_designs_group
data_designs_group_id=train_designs_group

N=2
for i in 1 2; do
  frac_part="$i/$N"
  for action_space_id in extended; do
    cmd="python ./core/algos/imitation_learning/behavioural_cloning/main_test_policy_simple.py \
                --designs_group_id $designs_group_id \
                --data_designs_group_id $data_designs_group_id \
                --frac_part $frac_part \
                --seq_length $seq_length \
                --mapping $mapping \
                --action_space_id $action_space_id \
                --ref_abc_seq $ref_abc_seq \
                --n_trials $n_trials \
                --data_n_trials $data_n_trials \
                --data_seed $data_seed $sample $log_transf $overwrite"
    echo $cmd
    $cmd
  done
done

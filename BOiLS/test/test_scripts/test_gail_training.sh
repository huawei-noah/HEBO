#!/bin/bash

mapping=fpga

ref_abc_seq=resyn2

seq_length=10

data_n_trials=1000

random_sampling_id='latin-hypercube'
seed=0
data_seed=0

log_transf=1
if (( log_transf == 0 )); then log_transf=''; else log_transf='--log_transf'; fi

n_envs=2

total_timesteps=20

overwrite=0
if (( overwrite == 0 )); then overwrite=''; else overwrite='--overwrite'; fi


designs_group_id=aux_test_designs_group
data_designs_group_id=train_designs_group

for action_space_id in extended; do
    cmd="python ./core/algos/imitation_learning/gail/gail_training.py \
                --designs_group_id $designs_group_id \
                --data_designs_group_id $data_designs_group_id \
                --seq_length $seq_length \
                --mapping $mapping \
                --action_space_id $action_space_id \
                --ref_abc_seq $ref_abc_seq \
                --data_n_trials $data_n_trials \
                --data_n_trials $data_n_trials \
                --data_seed $data_seed \
                --seed $seed \
                --n_envs $n_envs \
                --total_timesteps $total_timesteps $log_transf $overwrite"
    echo $cmd
    $cmd
done

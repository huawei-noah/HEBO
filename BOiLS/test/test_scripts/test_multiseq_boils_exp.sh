#!/bin/bash

mapping=fpga

ref_abc_seq=resyn2

seq_length=4

n_total_evals=4
n_initial=2
seed=0

n_parallel=10
n_universal_seqs=3
designs_group_id=aux_test_abc_graph

standardise=1
if (( standardise == 0 )); then standardise=''; else standardise='--standardise'; fi

ard=1
if (( ard == 0 )); then ard=''; else ard='--ard'; fi

acq=ei

overwrite=0
if (( overwrite == 0 )); then overwrite=''; else overwrite='--overwrite'; fi

device=-1
objective='both'

for action_space_id in extended; do
  cmd="python ./core/algos/bo/boils/main_multiseq_boils.py \
              --designs_group_id $designs_group_id \
              --n_parallel $n_parallel \
              --seq_length $seq_length \
              --n_universal_seqs $n_universal_seqs \
              --mapping $mapping \
              --action_space_id $action_space_id \
              --ref_abc_seq $ref_abc_seq \
              --n_total_evals $n_total_evals \
              --n_initial $n_initial \
              --device $device \
              $standardise $ard --acq $acq \
              --objective $objective \
              --seed $seed $overwrite"
  $cmd
  echo $cmd
done

#!/bin/bash

mapping=fpga

ref_abc_seq=resyn2

seq_length=10

n_total_evals=4
n_initial=2
seed=0

n_parallel=10
designs_group_id=aux_test_abc_graph

standardise=1
if (( standardise == 0 )); then standardise=''; else standardise='--standardise'; fi

ard=1
if (( ard == 0 )); then ard=''; else ard='--ard'; fi

acq=ei
kernel_type='s-bert-matern52'
#kernel_type='transformed_overlap'
embedder_path="sentence-bert-transf/data/fine_tuned/"

overwrite=0
if (( overwrite == 0 )); then overwrite=''; else overwrite='--overwrite'; fi

device=0
objective='both'
export CUDA_VISIBLE_DEVICES="0"
for action_space_id in extended; do
  cmd="python ./core/algos/bo/boils/main_multi_boils.py \
              --designs_group_id $designs_group_id \
              --n_parallel $n_parallel \
              --seq_length $seq_length \
              --mapping $mapping \
              --action_space_id $action_space_id \
              --ref_abc_seq $ref_abc_seq \
              --n_total_evals $n_total_evals \
              --n_initial $n_initial \
              --device $device \
              $standardise $ard --acq $acq \
              --kernel_type $kernel_type \
              --embedder_path $embedder_path \
              --objective $objective \
              --seed $seed $overwrite"
#  TOKENIZERS_PARALLELISM=0
  $cmd
  echo $cmd
done

#!/bin/bash

mapping=fpga

ref_abc_seq=resyn2
ref_abc_seq=init

seq_length=10

n_total_evals=1000
n_initial=20

n_parallel=1
#designs_group_id=hisi174_train
#designs_group_id=sqrt

standardise=1
if (( standardise == 0 )); then standardise=''; else standardise='--standardise'; fi

ard=1
if (( ard == 0 )); then ard=''; else ard='--ard'; fi

acq=ei
kernel_type='s-bert-matern52'
#kernel_type='transformed_overlap'
#embedder_path="sentence-bert-transf/data/fine_tuned/"
embedder_path="sentence-bert-transf-names/fine_tuned/"

name_embed=1
if (( name_embed == 0 )); then name_embed=''; else name_embed='--name_embed'; fi


lut_inputs=6
use_yosys=1

overwrite=0
if (( overwrite == 0 )); then overwrite=''; else overwrite='--overwrite'; fi

seed=0
device=0
objective='level'
export CUDA_VISIBLE_DEVICES="1"
action_space_id=extended

#for designs_group_id in div hyp log2 multiplier sin sqrt square; do
#for designs_group_id in adder hyp max div  ; do
#for designs_group_id in sin log2 multiplier  ; do
#for designs_group_id in sqrt square  ; do
for designs_group_id in hyp  ; do
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
              --lut_inputs $lut_inputs \
              --use_yosys $use_yosys \
              $standardise $ard --acq $acq \
              --kernel_type $kernel_type \
              --embedder_path $embedder_path $name_embed \
              --objective $objective \
              --seed $seed $overwrite"
   $cmd &
   echo $cmd
done

# scp -r /mnt/data/home/antoineg/eda-logic-synth/results/BO/fpga-6_seq-10_ref-init_act-extended/* antoineg@10.206.165.48:/mnt/data/antoineg/eda-logic-synth/results/BO/fpga-6_seq-10_ref-init_act-extended/
#!/bin/bash

# --- EDA setup ---

mapping=fpga       # parameter of the EDA mapping
lut_inputs=6       # parameter of the EDA mapping
use_yosys=1        # whether to use yosys-abc command or the abc_py package for the evaluation

# --- Exp setup ---

# Choose the logic synthesis sequence baseline
ref_abc_seq=resyn2 # the most-used one
#ref_abc_seq=init   # simply the empty sequence -> compare against doing nothing

seq_length=20 # the length of the sequence of operations (also the dimensionality of the problem)

n_parallel=1 # Number of sequences to evaluate in parallel (since we don't do batch BO, no need to set > 1)

objective='both'

# --- BO setup ---

n_total_evals=200 # Number of acquisitions in total (number of sequences that we will evaluate)
n_initial=20      # Number of initial random sequences to evaluate to build the first surrogate model

standardise=1
if ((standardise == 0)); then standardise=''; else standardise='--standardise'; fi

# deprecated
if ((seq_length < 0)); then
  ard=''
  failtol=1e6
  length_init_discrete_factor=1
else
  ard='--ard'
  failtol=40
  length_init_discrete_factor=".666"
fi

acq=ei
#kernel_type='s-bert-matern52'
kernel_type='transformed_overlap'
kernel_type='ssk'
#embedder_path="sentence-bert-transf/data/fine_tuned/"
embedder_path="sentence-bert-transf-names/fine_tuned/" # not important if the standard kernel is used

name_embed=0 # whether to use input warping taking operator names into account
if ((name_embed == 0)); then name_embed=''; else name_embed='--name_embed'; fi

# --- Run parameters ---

overwrite=0
if ((overwrite == 0)); then overwrite=''; else overwrite='--overwrite'; fi

#circuits=(hyp adder max bar sin square sqrt multiplier div log2)
#circuits=(adder)  # 46 - 1
circuits=(max adder sin)  # 46 - 2
circuits=(sin square sqrt multiplier log2 div hyp)  # 46 - 2
circuits=(sin square multiplier log2 div hyp)  # 46 - 2
circuits=(sqrt)  # 46 - 2

#for seed in {1..4}; do # can run for several seeds
for action_space_id in extended; do # can run for several action spaces (available operations in sequences)
  for designs_group_id in ${circuits[@]}; do
    i=1
    for objective in both; do
      for seed in {1..3}; do
#      for seed in 4; do
        taskset_start=$((i * 5))
        taskset_end=$((taskset_start + 4))
        i=$((i + 1))
        echo "$taskset_start-$taskset_end | $designs_group_id $objective $seed"

        device_to_export=$seed
        if ((device_to_export >= 0)); then device=0; else device=-1; fi
        if ((device_to_export >= 0)); then export CUDA_VISIBLE_DEVICES=$device_to_export; fi

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
                      --length_init_discrete_factor $length_init_discrete_factor \
                      --failtol $failtol \
                      --embedder_path $embedder_path $name_embed \
                      --objective $objective \
                      --seed $seed $overwrite"
        taskset -c $taskset_start-$taskset_end $cmd &
        echo $cmd
      done
    done
  wait
  done
done

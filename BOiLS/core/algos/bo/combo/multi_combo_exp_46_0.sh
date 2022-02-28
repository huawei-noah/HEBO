#!/bin/bash

# --- EDA setup ---

mapping=fpga       # parameter of the EDA mapping
lut_inputs=6       # parameter of the EDA mapping
use_yosys=1        # whether to use yosys-abc command or the abc_py package for the evaluation

# Choose the logic synthesis sequence baseline
ref_abc_seq=resyn2 # the most-used one
#ref_abc_seq=init   # simply the empty sequence -> compare against doing nothing

seq_length=20 # the length of the sequence of operations (also the dimensionality of the problem)

n_eval=200
n_initial=20

n_parallel=1

overwrite=0
if ((overwrite == 0)); then overwrite=''; else overwrite='--overwrite'; fi

device_to_export=0
if ((device_to_export >= 0)); then device=0; else device=-1; fi
if ((device_to_export >= 0)); then export CUDA_VISIBLE_DEVICES=$device_to_export; fi

#for designs_group_id in ; do # 46
#  for designs_group_id in hyp div log2 bar square adder multiplier sqrt sin max; do # 46
  for designs_group_id in max sqrt bar adder multiplier square; do # 46
    for action_space_id in extended; do
      i=0
      for seed in {0..4}; do
        for objective in both; do
          taskset_start=$((i * 4))
          taskset_end=$((taskset_start + 3))
          i=$((i + 1))
          echo "$taskset_start-$taskset_end | $designs_group_id $objective $seed"

          cmd="python ./core/algos/bo/combo/main_multi_combo.py \
                  --designs_group_id $designs_group_id \
                  --n_parallel $n_parallel \
                  --seq_length $seq_length \
                  --mapping $mapping \
                  --lut_inputs $lut_inputs \
                  --use_yosys $use_yosys
                  --action_space_id $action_space_id \
                  --ref_abc_seq $ref_abc_seq \
                  --n_eval $n_eval \
                  --n_initial $n_initial \
                  --device $device \
                  --objective $objective \
                  --seed $seed $overwrite"
        echo $cmd
        taskset -c $taskset_start-$taskset_end $cmd &
        echo $cmd
      done
    done
    wait
  done
done

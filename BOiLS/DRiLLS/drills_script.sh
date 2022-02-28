#!/bin/bash

iterations=20
episodes=1000
#bar max adder log2 sin  square sqrt  div multiplier hyp

lr=.01
mapping=fpga
lut_inputs=6
ref_abc_seq=resyn2

rec=""

#seed="0 1 2 3 4"

#for design in hyp div log2 multiplier bar sin sqrt square max adder; do #
#for design in adder max sin sqrt square; do # 23 - 0
#for design in log2 div multiplier hyp bar; do # 23 - 3
#for design in multiplier bar; do # 48 - 0
#for design in multiplier bar hyp; do #
  for action_space_id in extended; do
    for objective in both; do
      for algo in a2c; do
        for seed in {0..4}; do
          #    for action_space_id in extended; do
          #    for action_space_id in standard; do
          #        for design in bar sin sqrt square; do #
          #      for design in max adder; do #
          cmd="python ./DRiLLS/drills.py --mode train -e $episodes -i $iterations -d $design -a $algo \
                                           --ref_abc_seq $ref_abc_seq --mapping $mapping \
                                           --action_space_id $action_space_id \
                                           --seed $seed --objective $objective --lut_inputs $lut_inputs --lr $lr"
          echo $cmd
          $cmd &
        done
      done
    wait
    done
  done
done

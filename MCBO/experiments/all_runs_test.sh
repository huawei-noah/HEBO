#!/bin/bash

MAX_NUM_ITER=30
BO_N_INIT=4
RESULT_DIR="./test_results/"
VERBOSE=2
# --- Combinatorial exps ---

SEEDS="0"
ABSOLUT_EXE="./libs/Absolut/src/AbsolutNoLib"
for task in ackley aig_optimization antibody_design mig_optimization pest rna_inverse_fold; do
  for acq_opt in ls is sa ga; do
    for tr in "basic" "none"; do
      acq_func="ei"
      for model in gp_o gp_to gp_ssk gp_diff gp_hed; do
        opt_id="${model}__${acq_opt}__${acq_func}__${tr}"
        cmd="python ./experiments/run_task_exps.py --device_id 0 --absolut_dir $ABSOLUT_EXE --task_id $task --optimizers_ids $opt_id --seeds $SEEDS --max_num_iter $MAX_NUM_ITER --bo_n_init $BO_N_INIT --result_dir $RESULT_DIR --verbose $VERBOSE"
        echo $cmd
        $cmd
      done

      acq_func="ts"
      model=lr_sparse_hs
      opt_id="${model}__${acq_opt}__${acq_func}__${tr}"
      cmd="python ./experiments/run_task_exps.py --device_id 0 --absolut_dir $ABSOLUT_EXE --task_id $task --optimizers_ids $opt_id --seeds $SEEDS --max_num_iter $MAX_NUM_ITER --bo_n_init $BO_N_INIT --result_dir $RESULT_DIR --verbose $VERBOSE"
      echo $cmd
      $cmd
    done
  done

  for opt_id in ga sa rs hc; do
    cmd="python ./experiments/run_task_exps.py --device_id 0 --absolut_dir $ABSOLUT_EXE --task_id $task --optimizers_ids $opt_id --seeds $SEEDS --max_num_iter $MAX_NUM_ITER --bo_n_init $BO_N_INIT --result_dir $RESULT_DIR --verbose $VERBOSE"
    echo $cmd
    $cmd
  done
done

# --- Mixed exps ---

SEEDS="0"
for task in ackley-53 xgboost_opt aig_optimization_hyp svm_opt; do
  for acq_opt in mab is sa ga; do
    for tr in "basic" "none"; do
      acq_func="ei"
      for model in gp_o gp_to gp_hed; do
        opt_id="${model}__${acq_opt}__${acq_func}__${tr}"
        cmd="python ./experiments/run_task_exps.py --device_id 0 --task_id $task --optimizers_ids $opt_id --seeds $SEEDS --max_num_iter $MAX_NUM_ITER --bo_n_init $BO_N_INIT --result_dir $RESULT_DIR --verbose $VERBOSE"
        echo $cmd
        $cmd
      done
    done
  done

  for opt_id in ga sa rs hc; do
    cmd="python ./experiments/run_task_exps.py --device_id 0 --task_id $task --optimizers_ids $opt_id --seeds $SEEDS --max_num_iter $MAX_NUM_ITER --bo_n_init $BO_N_INIT --result_dir $RESULT_DIR --verbose $VERBOSE"
    echo $cmd
    $cmd
  done
done

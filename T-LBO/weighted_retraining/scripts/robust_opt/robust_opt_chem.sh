#!/bin/bash

# Meta flags
gpu="--gpu" # change to "" if no GPU is to be used
train_data_path="weighted_retraining/data/chem/zinc/orig_model/tensors_train"
val_data_path="weighted_retraining/data/chem/zinc/orig_model/tensors_val"
vocab_file_path="weighted_retraining/data/chem/zinc/orig_model/vocab.txt"
property_file_path="weighted_retraining/data/chem/zinc/orig_model/pen_logP_all.pkl"

# Weighted retraining parameters
k="1e-3"
r=50
weight_type="rank"
lso_strategy="opt"

samples_per_model=0

#-- Choose dimension of the latent space --#
latent_dim=56

#-- Choose acquisition function --#
acq_ind=0
acq_func_ids=(
  'ExpectedImprovement'
  'ErrorAwareEI'
  'ErrorAwareUCB')
gamma=1
eta=10
beta=2
acq_func_kwargs_s=(
  "{}"
  "{'gamma':$gamma,'eta':$eta}"
  "{'gamma':$gamma,'eta':$eta,'beta':$beta}")
acq_func_id=${acq_func_ids[$acq_ind]}
acq_func_kwargs=${acq_func_kwargs_s[$acq_ind]}
acq_func_opt_kwargs_s=("{'batch_limit':25}" "{'batch_limit':25,'maxiter':500}")
acq_func_opt_kwargs=${acq_func_opt_kwargs_s[$acq_ind]}

#-- Choose whether to use target prediction --#
predict_target=0
beta_target_pred_loss=10
target_predictor_hdims='[128,128]'
if ((predict_target == 0)); then predict_target=''; else predict_target='--predict_target'; fi

#-- Choose the metric loss you want to use -- (default contrastive: 2, default triplet: 3) --#
metric_loss_ind=3
metric_losses=('' 'contrastive' 'contrastive' 'triplet' 'triplet' 'triplet' 'log_ratio')
metric_loss_kws=("" "{'threshold':.05,'hard':True}" "{'threshold':.1}" "{'threshold':.1}" "{'threshold':.2,'soft':True}" "{'threshold':.1,'soft':True,'eta':0.05}" "{}")
beta_metric_loss_s=(1 1 1 1 1 1 1)
if ((metric_loss_ind == 0)); then metric_loss=''; else metric_loss="--metric_loss ${metric_losses[$metric_loss_ind]}"; fi
if ((metric_loss_ind == 0)); then metric_loss_kw=''; else metric_loss_kw="--metric_loss_kw ${metric_loss_kws[$metric_loss_ind]}"; fi
beta_metric_loss="${beta_metric_loss_s[$metric_loss_ind]}"

#-- Choose the pretrained model to start from (0: SLBO-Zero | 1: any semi-supervised exps or LSO | 3-4-5-6: specific metric leanring / target prediction) --#
pretrained_model_id_ind=1
pretrained_model_ids=('no-pretrained' 'vanilla' 'contrastive' 'triplet' 'pred_y' 'log_ratio')
pretrained_model_id=${pretrained_model_ids[$pretrained_model_id_ind]}
pretrained_model_file="./weighted_retraining/assets/pretrained_models/chem_$pretrained_model_id/chem.ckpt"

#-- Choose whether to use the pretrained model (0: SLBO-Zero | 1: all other cases) --#
use_pretrained=1
if ((use_pretrained == 0)); then lr='--lr 0.001'; else lr=''; fi
if ((use_pretrained == 0)); then use_pretrained=''; else use_pretrained='--use_pretrained'; fi

n_init_retrain_epochs_s=(15 1 0 0 0 0 0 0 0) # one init_retrain_epoch when metric learning is not used
n_init_retrain_epochs="${n_init_retrain_epochs_s[$pretrained_model_id_ind]}"

#-- Choose semi-supervised or fully-supervised setup --#
semi_supervised=0
n_init_bo_points=2000
if ((semi_supervised == 0)); then n_init_bo_points=''; else n_init_bo_points="--n_init_bo_points ${n_init_bo_points}"; fi
if ((semi_supervised == 0)); then n_retrain_epochs='0.1'; else n_retrain_epochs='1'; fi
if ((semi_supervised == 0)); then query_budget=500; else query_budget=1000; fi
if ((semi_supervised == 0)); then batch_size=128; else batch_size=32; fi
if ((semi_supervised == 0)); then semi_supervised=''; else semi_supervised='--semi_supervised'; fi

# KL coef in ELBO
beta_final=0.001

#-- Choose whether to use BO or random search --#
lso_strategy_ind=0
lso_strategies=("opt" "random_search")
lso_strategy="${lso_strategies[$lso_strategy_ind]}"

#-- Fit variational GP on a subset of the visible dataset --#
n_best_points=2000
n_rand_points=8000

#-- Run on the following seeds (repeat so that it restarts - not from scratch - after a potential crash)
seed_array=(0 1 2 3 4 0 1 2 3 4)

#-- Choose on which GPU to run --#
export CUDA_VISIBLE_DEVICES=0

expt_index=0 # Track experiments
for seed in "${seed_array[@]}"; do

  # Increment experiment index
  expt_index=$((expt_index + 1))

  # Break loop if using slurm and it's not the right task
  if [[ -n "${SLURM_ARRAY_TASK_ID}" ]] && [[ "${SLURM_ARRAY_TASK_ID}" != "$expt_index" ]]; then
    continue
  fi

  # Echo info of task to be executed
  echo "r=${r} k=${k} seed=${seed}"

  # Run command
  cmd="python ./weighted_retraining/weighted_retraining/robust_opt_scripts/robust_opt_chem.py \
            --seed=$seed $gpu \
            --query_budget=$query_budget \
            --retraining_frequency=$r \
            --pretrained_model_file=$pretrained_model_file \
            --pretrained_model_id $pretrained_model_id \
            --batch_size $batch_size \
            --lso_strategy=$lso_strategy \
            --train_path=$train_data_path \
            --val_path=$val_data_path \
            --vocab_file=$vocab_file_path \
            --property_file=$property_file_path \
            --n_retrain_epochs=$n_retrain_epochs \
            --latent_dim $latent_dim \
            $predict_target --beta_target_pred_loss $beta_target_pred_loss \
            --target_predictor_hdims $target_predictor_hdims $metric_loss $metric_loss_kw \
            --beta_metric_loss $beta_metric_loss --beta_final $beta_final \
            --n_init_retrain_epochs="$n_init_retrain_epochs" \
            --n_best_points=$n_best_points --n_rand_points=$n_rand_points \
            --n_inducing_points=500 $semi_supervised $n_init_bo_points \
            --samples_per_model $samples_per_model \
            --weight_type=$weight_type --rank_weight_k=$k \
            --acq-func-id $acq_func_id \
            --acq-func-kwargs $acq_func_kwargs \
            --acq-func-opt-kwargs $acq_func_opt_kwargs $use_pretrained $lr"
  $cmd
  echo $cmd

done

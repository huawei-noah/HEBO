#!/bin/bash

# for this task we use the BO script of the molecule design task to get the training done

# Meta flags
gpu="--gpu" # change to "" if no GPU is to be used
train_data_path="weighted_retraining/data/chem/zinc/orig_model/tensors_train"
val_data_path="weighted_retraining/data/chem/zinc/orig_model/tensors_val"
vocab_file_path="weighted_retraining/data/chem/zinc/orig_model/vocab.txt"
property_file_path="weighted_retraining/data/chem/zinc/orig_model/pen_logP_all.pkl"

# Weighted retraining parameters ---
k="1e-3"
weight_type="rank"

#-- Choose dimension of the latent space --#
latent_dim=56

#-- Choose whether to use target prediction --#
predict_target=0
beta_target_pred_loss=10
target_predictor_hdims='[128,128]'
if ((predict_target == 0)); then predict_target=''; else predict_target='--predict_target'; fi

#-- Choose the metric loss you want to use -- (default contrastive: 2, default triplet: 3) --#
metric_loss_ind=2
metric_losses=('' 'contrastive' 'contrastive' 'triplet' 'triplet' 'triplet' 'log_ratio')
metric_loss_kws=("" "{'threshold':.05,'hard':True}" "{'threshold':.1}" "{'threshold':.1,'soft':True}"
                  "{'threshold':.2,'soft':True}" "{'threshold':.1,'soft':True,'eta':0.05}" "{}")
save_model_ids=('vanilla' 'contrastive_hard_05' 'contrastive' 'triplet' 'triplet_02' 'triplet_eta_005' 'log_ratio')
if ((metric_loss_ind == 0)); then metric_loss=''; else metric_loss="--metric_loss ${metric_losses[$metric_loss_ind]}"; fi
if ((metric_loss_ind == 0)); then metric_loss_kw=''; else metric_loss_kw="--metric_loss_kw ${metric_loss_kws[$metric_loss_ind]}"; fi
batch_size_s=(32 256 256 1024 1024 1024 128)
batch_size=${batch_size_s[$metric_loss_ind]}
save_model_id=${save_model_ids[$metric_loss_ind]}
save_model_path="./weighted_retraining/assets/pretrained_models/chem_$save_model_id/chem.ckpt"
beta_metric_loss=1

#-- Start from pretrained vanilla JTVAE
pretrained_model_id='vanilla'
pretrained_model_file="./weighted_retraining/assets/pretrained_models/chem_$pretrained_model_id/chem.ckpt"

#-- Choose number of training epochs
n_init_retrain_epochs=.1

# KL coef in ELBO
beta_final=0.001

#-- Run on the following seeds (repeat so that it restarts - not from scratch - after a potential crash)
seed_array=(0)

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
            --query_budget=1 \
            --retraining_frequency=50 \
            --pretrained_model_file=$pretrained_model_file \
            --pretrained_model_id $pretrained_model_id \
            --batch_size $batch_size \
            --lso_strategy=opt \
            --train_path=$train_data_path \
            --val_path=$val_data_path \
            --vocab_file=$vocab_file_path \
            --property_file=$property_file_path \
            --n_retrain_epochs=.1 \
            --latent_dim $latent_dim \
            $predict_target --beta_target_pred_loss $beta_target_pred_loss \
            --target_predictor_hdims $target_predictor_hdims $metric_loss $metric_loss_kw \
            --beta_metric_loss $beta_metric_loss --beta_final $beta_final \
            --n_init_retrain_epochs="$n_init_retrain_epochs" \
            --n_best_points=2000 --n_rand_points=8000 \
            --n_inducing_points=500 --save_model_path $save_model_path --train-only \
            --weight_type=$weight_type --rank_weight_k=$k --use_pretrained"
  $cmd
  echo $cmd

done

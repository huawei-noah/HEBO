#!/bin/bash

seed=0

#-- Dataset composition --#
ignore_percentile=65
good_percentile=5
data_seed=0

#-- Choose dimension of the latent space --#
latent_dim=25

#-- Choose whether to use target prediction --#
predict_target=0
beta_target_pred_loss=1
target_predictor_hdims='[128,128]'
if ((predict_target == 0)); then predict_target=''; else predict_target='--predict_target'; fi

#-- Choose the metric loss you want to use --#
metric_loss_ind=3
metric_losses=('' 'contrastive' 'contrastive' 'contrastive' 'contrastive' 'triplet' 'triplet' 'log_ratio')
metric_loss_kws=("" "{'threshold':.1}" "{'threshold':.1,'hard':True}" "{'threshold':.05,'hard':True}"
                 "{'threshold':.2,'hard':True}" "{'threshold':.1,'soft':True}"
                  "{'threshold':.1,'soft':True,'eta':0.05}" "{}")
beta_metric_loss_s=(1 10 10 10 10 10 10 1)
batch_size_s=(256 256 256 256 256 256 256 128)
max_epochs_s=(300 300 300 300 300 300 300 150)
if ((metric_loss_ind == 0)); then metric_loss=''; else metric_loss="--metric_loss ${metric_losses[$metric_loss_ind]}"; fi
if ((metric_loss_ind == 0)); then metric_loss_kw=''; else metric_loss_kw="--metric_loss_kw ${metric_loss_kws[$metric_loss_ind]}"; fi
beta_metric_loss="${beta_metric_loss_s[$metric_loss_ind]}"
batch_size="${batch_size_s[$metric_loss_ind]}"
max_epochs="${max_epochs_s[$metric_loss_ind]}"

#-- Choose on which GPU to run --#
cuda=0

# Train expr VAE
k="1e-3"

cmd="python weighted_retraining/weighted_retraining/partial_train_scripts/partial_train_expr.py \
  --seed=$seed  \
  --cuda=$cuda --batch_size $batch_size \
  --latent_dim=$latent_dim \
  --dataset_path=weighted_retraining/data/expr \
  --property_key=scores \
  --max_epochs=$max_epochs \
  --beta_final=.04 --beta_start=1e-6 \
  --beta_warmup=500 --beta_step=1.1 --beta_step_freq=10 \
  --weight_type rank --rank_weight_k $k --data_seed $data_seed \
  --ignore_percentile $ignore_percentile --good_percentile $good_percentile \
  $predict_target --target_predictor_hdims $target_predictor_hdims \
  $metric_loss $metric_loss_kw --beta_metric_loss=$beta_metric_loss \
  --beta_target_pred_loss=$beta_target_pred_loss
"
echo $cmd
$cmd
echo $cmd

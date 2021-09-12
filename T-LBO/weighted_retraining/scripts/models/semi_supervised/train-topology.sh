#!/bin/bash

seed=0

#-- Choose whether to use target prediction --#
predict_target=0
beta_target_pred_loss=10
target_predictor_hdims='[128,128]'
if ((predict_target == 0)); then predict_target=''; else predict_target='--predict_target'; fi

#-- Choose the metric loss you want to use --#
metric_loss_ind=4
metric_losses=('' 'contrastive' 'contrastive' 'triplet' 'triplet')
metric_loss_kws=("" "{'threshold':.1}" "{'threshold':.1,'hard':True}" "{'threshold':.1,'soft':True}" "{'threshold':.1,'soft':True,'eta':0.05}")
beta_metric_loss_s=(1 1 1 1 1)
if ((metric_loss_ind == 0)); then metric_loss=''; else metric_loss="--metric_loss ${metric_losses[$metric_loss_ind]}"; fi
if ((metric_loss_ind == 0)); then metric_loss_kw=''; else metric_loss_kw="--metric_loss_kw ${metric_loss_kws[$metric_loss_ind]}"; fi
beta_metric_loss="${beta_metric_loss_s[$metric_loss_ind]}"

use_binary_data=1
if (( use_binary_data == 1 )); then use_binary_data='--use_binary_data'; else use_binary_data=''; fi

#-- Choose dimension of the latent space --#
latent_dim=20

#-- For how many epochs do you want to train the model? --#
max_epochs=300

#-- Choose on which GPU to run --#
cuda=0

# Train topo VAE
python weighted_retraining/weighted_retraining/partial_train_scripts/partial_train_topology.py \
--seed=$seed \
--latent_dim=$latent_dim \
--property_key=scores \
--max_epochs=$max_epochs \
--beta_final=1e-4 --beta_start=1e-6 \
--beta_warmup=1000 --beta_step=1.1 --beta_step_freq=10 \
--batch_size=1024 \
--cuda=$cuda --weight_type rank --rank_weight_k 'inf' \
$predict_target --target_predictor_hdims $target_predictor_hdims --beta_target_pred_loss $beta_target_pred_loss \
$metric_loss $metric_loss_kw --beta_metric_loss $beta_metric_loss \
$use_binary_data

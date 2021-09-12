#!/bin/bash

#-- Common parameters for expression task -#

dataset_path="weighted_retraining/data/expr"
property_key='scores'
weight_type="rank"

data_seed=0
n_data=100000
use_full_data_for_gp="" # "--use_full_data_for_gp"
n_decode_attempts=20
ignore_percentile=65
good_percentile=5
samples_per_model=0
n_retrain_epochs=1
test_dir=""
use_test_set="" # "--use_test_set"

#-- Fit variational GP on a subset of the visible dataset --#
n_inducing_points=500
n_best_points=2500
n_rand_points=500

#################################### SETUP ####################################

# if several trainings have been launched with similar setups, pytorch_lightning have saved models in directories
# `version_0`, `version_1`,... so the model version shall be specified here (0 by default)
version=0
r=50     # retrain VAE every $r steps
k="1e-3" # factor for weighted retraining

#-- Choose covariance function for the GP --#
covar_name_ind=1
covar_names=('matern-5/2' 'rbf')
covar_name=${covar_names[$covar_name_ind]}

#-- Choose dimension of the latent space --#
latent_dim=25

#-- Choose acquisition function --#
acq_ind=1
acq_func_ids=(
  'ExpectedImprovement'
  'ErrorAwareEI'
  'ErrorAwareUCB'
)
gamma=1
eps=10
beta=2
config="ratio"
acq_func_kwargs_s=(
  "{}"
  "{'gamma':$gamma,'eps':$eps,'configuration':'$config'}"
  "{'gamma':$gamma,'eps':$eps,'beta':$beta,'configuration':'$config'}"
)
acq_func_id=${acq_func_ids[$acq_ind]}
acq_func_kwargs=${acq_func_kwargs_s[$acq_ind]}
acq_func_opt_kwargs_s=("{'batch_limit':100}" "{'batch_limit':100,'maxiter':500}")

cost_aware_gamma_sched_ind=0
cost_aware_gamma_scheds=(
  ''
  '--cost_aware_gamma_sched fixed'
  '--cost_aware_gamma_sched linear'
  '--cost_aware_gamma_sched exponential'
  '--cost_aware_gamma_sched reverse_exponential'
  '--cost_aware_gamma_sched post_obj_var'
  '--cost_aware_gamma_sched post_obj_inv_var'
  '--cost_aware_gamma_sched post_err_var'
  '--cost_aware_gamma_sched post_min_var'
  '--cost_aware_gamma_sched post_var_tradeoff'
  '--cost_aware_gamma_sched post_var_inv_tradeoff')
cost_aware_gamma_sched=${cost_aware_gamma_scheds[$cost_aware_gamma_sched_ind]}

test_gp_error_fit='' #'--test_gp_error_fit'

estimate_rec_error=0
if ((estimate_rec_error == 0)); then estimate_rec_error=''; else estimate_rec_error='--estimate_rec_error'; fi

#-- Choose whether to use input warping --#
input_wp=0
if ((input_wp == 1)); then acq_func_opt_kwargs="{'batch_limit':50,'maxiter':500,'clip_gradient':1,'clip_value':10.,'jitter':1e-3}"; else acq_func_opt_kwargs=${acq_func_opt_kwargs_s[$acq_ind]}; fi
if ((input_wp == 0)); then input_wp=''; else input_wp='--input_wp'; fi

#-- Choose whether to use target prediction --#
predict_target=0
beta_target_pred_loss=1
target_predictor_hdims='[128,128]'
if ((predict_target == 0)); then predict_target=''; else predict_target='--predict_target'; fi

#-- Choose the metric loss you want to use -- (default contrastive: 1, default triplet: 5) --#
metric_loss_ind=3
metric_losses=('' 'contrastive' 'contrastive' 'contrastive' 'contrastive' 'triplet' 'triplet' 'triplet' 'log_ratio')
metric_loss_kws=("" "{'threshold':.1}" "{'threshold':.1,'hard':True}" "{'threshold':.05,'hard':True}"
  "{'threshold':.2,'hard':True}" "{'threshold':.1}" "{'threshold':.1,'soft':True}"
  "{'threshold':.1,'soft':True,'eta':0.05}" "{}")
beta_metric_loss_s=(1 10 10 10 10 10 10 10 1)
batch_size_s=(256 256 256 256 256 256 256 256 128)
if ((metric_loss_ind == 0)); then metric_loss=''; else metric_loss="--metric_loss ${metric_losses[$metric_loss_ind]}"; fi
if ((metric_loss_ind == 0)); then metric_loss_kw=''; else metric_loss_kw="--metric_loss_kw ${metric_loss_kws[$metric_loss_ind]}"; fi
beta_metric_loss="${beta_metric_loss_s[$metric_loss_ind]}"
batch_size="${batch_size_s[$metric_loss_ind]}"

#-- Choose whether to use the pretrained model (0: SLBO-Zero | 1: all other cases) --#
use_pretrained=0
if ((use_pretrained == 0)); then n_init_retrain_epochs=600; else n_init_retrain_epochs=1; fi
if ((use_pretrained == 0)); then lr=1e-2; else lr=1e-3; fi
if ((use_pretrained == 0)); then use_pretrained=''; else use_pretrained='--use_pretrained'; fi

# KL coef in ELBO
beta_final=0.04 # same as in train-expr-pt.sh

#-- Choose semi-supervised or fully-supervised setup --#
semi_supervised=0
n_init_bo_points=105
if ((semi_supervised == 0)); then n_init_bo_points=''; else n_init_bo_points="--n_init_bo_points ${n_init_bo_points}"; fi
if ((semi_supervised == 0)); then query_budget=500; else query_budget=1000; fi
if ((semi_supervised == 0)); then semi_supervised=''; else semi_supervised='--semi_supervised'; fi

#-- Choose whether to use BO or random search --#
lso_strategy_ind=0
lso_strategies=("opt" "random_search")
lso_strategy="${lso_strategies[$lso_strategy_ind]}"

#-- If random search is chosen for optimisation, choose standard (0) or qMC (1)
random_search_type_ind=1
random_search_types=("" "--random_search_type sobol")
random_search_type="${random_search_types[$random_search_type_ind]}"

#-- How many epochs have been used for VAE training --#
training_max_epochs=300

#-- Run on the following seeds (repeat so that it restarts - not from scratch - after a potential crash)
seed_array=(0 1 2 3 4 0 1 2 3 4)

#-- Choose on which GPU to run --#
cuda=0

##############################################################################

expt_index=0 # Track experiments
for seed in "${seed_array[@]}"; do
  # Increment experiment index
  expt_index=$((expt_index + 1))

  # Break loop if using slurm and it's not the right task
  if [[ -n "${SLURM_ARRAY_TASK_ID}" ]] && [[ "${SLURM_ARRAY_TASK_ID}" != "$expt_index" ]]; then
    continue
  fi

  echo "r=${r} k=${k} seed=${seed}"

  # Run command
  cmd="python weighted_retraining/weighted_retraining/robust_opt_scripts/robust_opt_expr.py \
        --seed=$seed $gpu \
        --query_budget=$query_budget \
        --retraining_frequency=$r \
        --version $version \
        --dataset_path=$dataset_path --property_key $property_key \
        --lso_strategy=$lso_strategy \
        $random_search_type          --n_retrain_epochs=$n_retrain_epochs \
        --n_init_retrain_epochs=$n_init_retrain_epochs \
        --n_best_points=$n_best_points --n_rand_points=$n_rand_points $use_full_data_for_gp \
        --n_inducing_points=$n_inducing_points \
        --weight_type=$weight_type --rank_weight_k=$k \
        --samples_per_model=$samples_per_model \
        --n_decode_attempts=$n_decode_attempts \
        --ignore_percentile=$ignore_percentile \
        --good_percentile $good_percentile \
        --batch_size $batch_size \
        --covar-name $covar_name \
        --acq-func-id $acq_func_id \
        --acq-func-kwargs $acq_func_kwargs \
        --acq-func-opt-kwargs $acq_func_opt_kwargs \
        --beta_metric_loss $beta_metric_loss --beta_final $beta_final \
        --beta_target_pred_loss $beta_target_pred_loss \
        --data_seed $data_seed $estimate_rec_error \
        --training_max_epochs $training_max_epochs \
        --cuda $cuda $use_test_set \
        $input_wp $raw_initial_samples \
        $predict_target \
        --target_predictor_hdims $target_predictor_hdims $metric_loss \
        $metric_loss_kw \
        $test_gp_error_fit $cost_aware_gamma_sched \
        --latent_dim $latent_dim $semi_supervised $n_init_bo_points --lr $lr $semi_supervised $n_init_bo_points $use_pretrained"
  $cmd
  echo $cmd
done

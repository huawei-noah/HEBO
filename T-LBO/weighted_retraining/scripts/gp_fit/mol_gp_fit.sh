# Meta flags
gpu="--gpu" # change to "" if no GPU is to be used
train_data_path="weighted_retraining/data/chem/zinc/orig_model/tensors_train"
val_data_path="weighted_retraining/data/chem/zinc/orig_model/tensors_val"
vocab_file_path="weighted_retraining/data/chem/zinc/orig_model/vocab.txt"
property_file_path="weighted_retraining/data/chem/zinc/orig_model/pen_logP_all.pkl"

# Weighted retraining parameters
k="1e-3"
weight_type="rank"

latent_dim=56

predict_target=0
beta_target_pred_loss=10
target_predictor_hdims='[128,128]'
if ((predict_target == 0)); then predict_target=''; else predict_target='--predict_target'; fi

metric_loss_ind=2
metric_losses=('' 'contrastive' 'triplet' 'triplet')
metric_loss_kws=("" "{'threshold':.1}" "{'threshold':.1,'soft':True}" "{'threshold':.1,'soft':True,'eta':0.05}")
beta_metric_loss_s=(1 1 1 1)
if ((metric_loss_ind == 0)); then metric_loss=''; else metric_loss="--metric_loss ${metric_losses[$metric_loss_ind]}"; fi
if ((metric_loss_ind == 0)); then metric_loss_kw=''; else metric_loss_kw="--metric_loss_kw ${metric_loss_kws[$metric_loss_ind]}"; fi
beta_metric_loss="${beta_metric_loss_s[$metric_loss_ind]}"

pretrained_model_id_ind=3
pretrained_model_ids=('no-pretrained' 'vanilla_weight' 'contrastive' 'triplet' 'pred_y' 'pred_y_triplet')
pretrained_model_id=${pretrained_model_ids[$pretrained_model_id_ind]}
pretrained_model_file="pretrained_models/chem_$pretrained_model_id/chem.ckpt"

use_pretrained=1
if ((use_pretrained == 0)); then use_pretrained=''; else use_pretrained='--use_pretrained'; fi

n_init_retrain_epochs_s=(10 0 0 0 0 0) # one init_retrain_epoch when metric learning is used
n_init_retrain_epochs="${n_init_retrain_epochs_s[$pretrained_model_id_ind]}"

semi_supervised=0
n_init_bo_points=2000
if ((semi_supervised == 0)); then n_init_bo_points=''; else n_init_bo_points="--n_init_bo_points ${n_init_bo_points}"; fi
if ((semi_supervised == 0)); then n_retrain_epochs='0.1'; else n_retrain_epochs='1'; fi
if ((semi_supervised == 0)); then query_budget=500; else query_budget=1000; fi
if ((semi_supervised == 0)); then batch_size=128; else batch_size=32; fi
if ((semi_supervised == 0)); then semi_supervised=''; else semi_supervised='--semi_supervised'; fi

beta_final=0.001

n_best_points=2000
n_rand_points=8000
n_test_points=2500

seed_array=(0 1 2 3 4 5 6 7 8 9)

export CUDA_VISIBLE_DEVICES=0

expt_index=0 # Track experiments

for use_decoded in 0 1; do
  if ((use_decoded == 0)); then use_decoded=''; else use_decoded='--use_decoded'; fi

  for seed in "${seed_array[@]}"; do

    # Increment experiment index
    expt_index=$((expt_index + 1))

    # Break loop if using slurm and it's not the right task
    if [[ -n "${SLURM_ARRAY_TASK_ID}" ]] && [[ "${SLURM_ARRAY_TASK_ID}" != "$expt_index" ]]; then
      continue
    fi

    # Echo info of task to be executed
    echo "k=${k} n_test_points=${n_test_points} use_decoded=${use_decoded}"

    # Run command
    cmd="python ./weighted_retraining/test/test_gp_fit.py \
              --seed=$seed $gpu \
              --pretrained_model_file=$pretrained_model_file \
              --pretrained_model_id $pretrained_model_id \
              --batch_size $batch_size \
              --train_path=$train_data_path \
              --val_path=$val_data_path \
              --vocab_file=$vocab_file_path \
              --property_file=$property_file_path \
              --latent_dim $latent_dim \
              $predict_target --beta_target_pred_loss $beta_target_pred_loss \
              --target_predictor_hdims $target_predictor_hdims $metric_loss $metric_loss_kw \
              --beta_metric_loss $beta_metric_loss --beta_final $beta_final \
              --n_init_retrain_epochs="$n_init_retrain_epochs" \
              --n_best_points=$n_best_points --n_rand_points=$n_rand_points \
              --n_inducing_points=500 $semi_supervised $n_init_bo_points \
              --weight_type=$weight_type --rank_weight_k=$k $use_pretrained \
              --n_test_points $n_test_points $use_decoded"
    $cmd
    echo $cmd

  done
done
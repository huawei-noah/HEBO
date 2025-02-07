read -p "CUDA_VISIBLE_DEVICES: " DEVICE
N_DEVICES="${DEVICE//[^[:digit:]]/}"

read -p "--data_path: " DATA_PATH
read -p "--model_path: " MODEL_PATH
read -p "--optim ['sft', 'dpo']: " OPTIM

DATA_PATH="/YOUR_BASE_PATH/datasets/${DATA_PATH}"
MODEL_PATH="/YOUR_BASE_PATH/models/${MODEL_PATH}"

if [ ${OPTIM} == "sft" ]; then
    read -p "--top_p: " TOP_P
    TOP_P="--top_p ${TOP_P}"
else
    read -p "--task ['qvs', 'pvf', 'all']: " TASK
    TASK="--task ${TASK}"
fi

read -p "--augment ['True', 'False']: " AUGMENT
read -p "--save_path: " SAVE_PATH


DISTRIBUTED_ARGS="--rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node=${#N_DEVICES}"

CUDA_VISIBLE_DEVICES=${DEVICE} torchrun ${DISTRIBUTED_ARGS} train.py \
    --data_path ${DATA_PATH} \
    --model_path ${MODEL_PATH} \
    --optim ${OPTIM} \
    ${TOP_P} \
    ${TASK} \
    --augment ${AUGMENT} \
    --ds_config "ds_zero.json" \
    --save_path "/YOUR_BASE_PATH/models/${SAVE_PATH}" \
    --seed 1

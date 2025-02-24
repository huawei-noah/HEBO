read -p "CUDA_VISIBLE_DEVICES: " DEVICE
N_DEVICES="${DEVICE//[^[:digit:]]/}"

read -p "--regen: " REGEN
read -p "--data_path: " DATA_PATH

if [ ${REGEN} == "True" ]; then
    read -p "--model_path: " MODEL_PATH

    if [[ $MODEL_PATH == *"StarCoder"* ]] || [[ $MODEL_PATH == *"CodeLlama"* ]]; then
        MODEL_PATH="--model_path /YOUR_BASE_PATH/models/${MODEL_PATH}"
    else
        MODEL_PATH="--model_path /YOUR_BASE_PATH/models/${MODEL_PATH}"
    fi

    read -p "--n_seq [1]: " N_SEQ
    read -p "--n_iter [1]: " N_ITER

    if [ ${N_SEQ} -gt 1 ]; then
        SAMPLE="True"
        read -p "--temp [1.0]: " TEMP
    fi
fi

read -p "--save_path: " SAVE_PATH


DISTRIBUTED_ARGS="--rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node=${#N_DEVICES}"

CUDA_VISIBLE_DEVICES=${DEVICE} torchrun ${DISTRIBUTED_ARGS} infer.py \
    --eval_mode "True" \
    --regen ${REGEN} \
    --data_path "/YOUR_BASE_PATH/datasets/${DATA_PATH}" \
    ${MODEL_PATH} \
    --n_seq ${N_SEQ:=1} \
    --n_iter ${N_ITER:=1} \
    --sample ${SAMPLE:="False"} \
    --temp ${TEMP:=1.0} \
    --save_path "/YOUR_BASE_PATH/tests${SAVE_PATH}/${DATA_PATH}" \
    --seed 1

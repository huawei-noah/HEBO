read -p "--data_path: " DATA_PATH
read -p "--save_path [${DATA_PATH}]: " SAVE_PATH

if [ -z "${SAVE_PATH}" ]; then
    SAVE_PATH=${DATA_PATH}
fi


torchrun merge.py \
    --data_path "/YOUR_BASE_PATH/samples/${DATA_PATH}" \
    --save_path "/YOUR_BASE_PATH/datasets/${SAVE_PATH}"

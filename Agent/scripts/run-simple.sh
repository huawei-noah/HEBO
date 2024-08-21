#!/bin/bash
set -e

# CONFIGURATIONS
PYTHON_INTERPRETER="python3"
LLM_NAME="qwen2-72b"
LLM_HOST="llm_playground"
SEEDS=1
NUM_WORKERS=2
REFLECTION=$1
RETRY="1"

if [[ -z "$REFLECTION" ]]; then
  REFLECTION_STRATEGY=null
else
  REFLECTION_STRATEGY="$REFLECTION"
fi

# RUN ON ALL TASKS
TASK_LIST=(
  "higgs-boson-simple"
  "mercedes-benz-greener-manufacturing-simple"
)

#TASKS_STRING=$(IFS=,; echo "${TASK_LIST[*]}")
TASKS_STRING=$(echo "${TASK_LIST[*]}")

if [[ -z "$RETRY" ]]; then
  $PYTHON_INTERPRETER third_party/hyperopt/run_experiments_parallel.py \
    --llm-name $LLM_NAME \
    --llm-host $LLM_HOST \
    --seeds $SEEDS \
    --num-workers $NUM_WORKERS \
    --tasks $TASKS_STRING
else
    $PYTHON_INTERPRETER third_party/hyperopt/run_experiments_parallel.py \
    --llm-name $LLM_NAME \
    --llm-host $LLM_HOST \
    --seeds $SEEDS \
    --num-workers $NUM_WORKERS \
    --tasks $TASKS_STRING \
    --retry-empty-seeds
fi

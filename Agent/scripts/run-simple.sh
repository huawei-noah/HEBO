#!/bin/bash
set -e

# CONFIGURATIONS
PYTHON_INTERPRETER="python3"
LLM_NAME="qwen2-72b"
LLM_HOST="llm_playground"
SEEDS=10
NUM_WORKERS=10

# Check for command line arguments
REFLECTION_STRATEGY=${1:-}  # Default to empty if not provided
RETRY=${2:-}                # Default to empty if not provided

# RUN ON ALL TASKS
TASK_LIST=(
#  "higgs-boson"
#  "abalone"
  "bank-churn"
)

TASKS_STRING=$(echo "${TASK_LIST[*]}")

# Build the command
COMMAND="$PYTHON_INTERPRETER scripts/run_experiments_parallel.py \
--llm-name $LLM_NAME \
--llm-host $LLM_HOST \
--seeds $SEEDS \
--num-workers $NUM_WORKERS"

# Add reflection strategy if provided
if [ -n "$REFLECTION_STRATEGY" ]; then
  COMMAND+=" --reflection-strategy $REFLECTION_STRATEGY"
fi

# Add retry if provided
if [ -n "$RETRY" ]; then
  COMMAND+=" --retry-empty-seeds"
fi

# Add tasks
COMMAND+=" --tasks $TASKS_STRING"

# Execute the command
echo $COMMAND
eval $COMMAND
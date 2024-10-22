# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Sentiment Control (IMDB)

# The imdb dataset was obtained from https://raw.githubusercontent.com/rycolab/odpo/refs/heads/main/data/
# and processed with https://github.com/rycolab/odpo/blob/main/preference_datasets.py#L249

accelerate launch \
--config_file "./configs/acc_config.yaml" \
--num_processes=4 \
--gpu_ids="0,1,2,3" \
run_train.py "./configs/config_po.yaml" \
--output_dir="<output-directory>" \
--model_name_or_path="insub/gpt2-large-imdb-fine-tuned" \
--tokenizer_name_or_path="gpt2-large" \
--dataset_mixer="{'<imdb-dataset>': 1.0}" \
--pref_optim="mapo" \
--activation_hook="all" \
--activation_mapping="zn_rescale" \
--beta=0.8 \
--learning_rate=1e-6 \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=1 \
--gradient_accumulation_steps=16 \
--num_train_epochs=3


# Summarization (TL;DR)

# The openai/summarize_from_feedback dataset was used for experiments
# by processing it to include both subsets (axis & comparisons) into the
# standard PO format with fields "prompt", "chosen", "rejected"

accelerate launch \
--config_file "./configs/acc_config.yaml" \
--num_processes=4 \
--gpu_ids="0,1,2,3" \
run_train.py "./configs/config_po_lora.yaml" \
--output_dir="<output-directory>" \
--model_name_or_path="CarperAI/openai_summarize_tldr_sft" \
--tokenizer_name_or_path="EleutherAI/gpt-j-6b" \
--torch_dtype="float16" \
--dataset_mixer="{'<tldr-processed-dataset>': 1.0}" \
--pref_optim="mapo" \
--activation_hook="all" \
--activation_mapping="zn_rescale" \
--beta=0.8 \
--learning_rate=1e-4 \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=1 \
--gradient_accumulation_steps=64 \
--num_train_epochs=1


# Helpfulness & Harmlessness (Anthropic_HH)

accelerate launch \
--config_file "./configs/acc_config.yaml" \
--num_processes=4 \
--gpu_ids="0,1,2,3" \
run_train.py "./configs/config_po.yaml" \
--output_dir="<output-directory>" \
--model_name_or_path="<sft-model>" \
--dataset_mixer="{'Anthropic/hh-rlhf': 1.0}" \
--pref_optim="mapo" \
--activation_hook="all" \
--activation_mapping="zn_rescale" \
--beta=0.1 \
--learning_rate=1e-6 \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=1 \
--gradient_accumulation_steps=16 \
--num_train_epochs=3



# Text-to-Code Generation (MBPP)

# The dataset was constructed following the process described in 
# https://arxiv.org/pdf/2406.12502
# with fields "question" and "answers" where answers contains a list of "text" and "votes". 
# "votes" can be any float number indicating a pass solution
# "votes" with a value of "none" indicate a failed solution

accelerate launch \
--config_file "./configs/acc_config.yaml" \
--num_processes=2 \
--gpu_ids="0,1" \
run_train.py "./configs/config_po.yaml" \
--output_dir="<output-directory>" \
--model_name_or_path="bigcode/starcoderbase-1b" \
--tokenizer_name_or_path="bigcode/starcoderbase-1b" \
--torch_dtype="float16" \
--dataset_mixer="{'<mbpp-dataset>': 1.0}" \
--dataset_splits="train,validation" \
--pref_optim="sparse" \
--mask_model="simple_all" \
--rw_kl_independent=False \
--beta=0.8 \
--learning_rate=5e-7 \
--per_device_train_batch_size=2 \
--per_device_eval_batch_size=2 \
--gradient_accumulation_steps=8 \
--num_train_epochs=3
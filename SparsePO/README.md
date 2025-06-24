# SparsePO: Controlling Preference Alignment of LLMs via Sparse Token Masks

This is the official implementation for the paper
[SparsePO: Controlling Preference Alignment of LLMs via Sparse Token Masks](https://arxiv.org/pdf/2410.05102).

If you like this work and plan to use it, please cite as follows:
```html
@article{christopoulou2024sparsepo,
  title={SparsePO: Controlling Preference Alignment of LLMs via Sparse Token Masks},
  author={Christopoulou, Fenia and Cardenas, Ronald and Lampouras, Gerasimos and Bou-Ammar, Haitham and Wang, Jun},
  journal={arXiv preprint arXiv:2410.05102},
  year={2024}
}
```


## Environment

Setup the environment by simply installing the main repo:
```bash
git clone https://github.com/huawei-noah/HEBO/tree/master/SparsePO
cd SparsePO
pip install -e .
```

## Training

To train PO models we follow the recipe from the [alignment-handbook](https://github.com/huggingface/alignment-handbook).

We use existing supervised fine-tuned models for the following experiments:
- IMBD: [insub/gpt2-large-imdb-fine-tuned](https://huggingface.co/insub/gpt2-large-imdb-fine-tuned)
- TL;DR: [CarperAI/openai_summarize_tldr_sft](https://huggingface.co/CarperAI/openai_summarize_tldr_sft)
- Text-to-Code Generation: [bigcode/starcoderbase-1b](https://huggingface.co/bigcode/starcoderbase-1b)

For HH, we perform SFT as follows:
```python
accelerate launch \
  --config_file ./configs/acc_config.yaml \
  --num_processes=4 \
  --gpu_ids="0,1,2,3" \
  run_train.py \
  "./configs/config_sft.yaml" \
  --output_dir="output_dir" \
  --pref_optim="sft" \
  --learning_rate=1e-5 \
  --num_train_epochs=1 \
  --per_device_train_batch_size=16 \
  --per_device_eval_batch_size=16 \
  --gradient_accumulation_steps=16
```

We incorporate our method into the [HuggingFace TRL library](https://github.com/huggingface/trl), providing additional trainers, named `sparse` and `mapo` 
inside `src/trainers/`.
To run the PO experiments, follow the sample code in `exps.sh`. 
We provide a table with the required hyper-parameters for each experiments below (change accordingly):

| Dataset | PO            | Arguments | Effective BS |
|---------|---------------|-----------|:------------:|
| IMDB    | mapo          | --pref_optim="mapo" <br>--activation_hook="all"  <br>--activation_mapping="zn_rescale" <br>--beta=0.8 <br> --learning_rate=1e-6 <br>--num_train_epochs 3 | 64 |
| IMDB    | sparse-common | --pref_optim="sparse" <br>--mask_model="simple_all" <br>--rw_kl_independent=False <br>--beta=0.8 <br>--learning_rate=1e-6 <br>--num_train_epochs 3 |
| IMDB    | sparse-indp   | --pref_optim="sparse" <br>--mask_model="simple_all" <br>--rw_kl_independent=True <br>--beta=0.8 <br>--learning_rate=1e-6 <br>--num_train_epochs 3 |
| TL;DR   | mapo          | --pref_optim="mapo" <br>--activation_hook="all" <br>--activation_mapping="zn_rescale" <br>--beta=0.8 <br>--learning_rate=1e-4 <br>--num_train_epochs 1 | 256 |
| TL;DR   | sparse-common | --pref_optim="sparse" <br>--mask_model="simple_all" <br>--rw_kl_independent=False <br>--beta=0.8 <br>--learning_rate=1e-4 <br>--num_train_epochs 1 <br>--mask_weight_decay 0.01 |
| TL;DR   | sparse-indp   | --pref_optim="sparse" <br>--mask_model="simple_all" <br>--rw_kl_independent=True <br>--beta=0.8 <br>--learning_rate=1e-4 <br>--num_train_epochs 1 <br>--mask_weight_decay 0.01 |
| HH      | mapo          | --pref_optim="mapo" <br>--activation_hook="all" <br>--activation_mapping="zn_rescale" <br>--beta=0.1 <br>--learning_rate=1e-6 <br>--num_train_epochs 3 | 128 |
| HH      | sparse-common | --pref_optim="sparse" <br>--mask_model="simple_all" <br>--rw_kl_independent=False <br>--beta=0.1 <br>--learning_rate=5e-7 <br>--num_train_epochs 3 <br>--mask_weight_decay 0.01 <br>--l1_norm_param_u=0.001 <br>--l1_norm_param_d=0.001 |
| HH      | sparse-indp   | --pref_optim="sparse" <br>--mask_model="simple_all" <br>--rw_kl_independent=True <br>--beta=0.1 <br>--learning_rate=5e-7 <br>--num_train_epochs 3 |
| MBPP    | mapo          | --pref_optim="mapo" <br>--activation_hook="all"  <br>--activation_mapping="zn_rescale" <br>--learning_rate=5e-7 | 32 |
| MBPP    | sparse-common | --pref_optim="sparse" <br>--mask_model="simple_all" <br>--rw_kl_independent=False <br>--learning_rate=5e-7
| MBPP    | sparse-indp   | --pref_optim="sparse" <br>--mask_model="simple_all" <br>--rw_kl_independent=True <br>--learning_rate=5e-7


## Evaluation

Evaluation on each domain is performed as follows:

### Summarization (TL;DR)

To evaluate summarization models, we employ the following metrics on 100 instances from the TL;DR test set generating 5 samples for each prompt using nucleus sampling with `p = 0.5` and temperatures `[0, 0.25, 0.50, 0.75, 1.0]`:
- ROUGE: [ROUGE from HF](https://github.com/huggingface/evaluate/blob/main/metrics/rouge)
- BERTScore: [BERTScore from HF](https://github.com/huggingface/evaluate/tree/main/metrics/bertscore)
- self-BLUE: [SacreBLUE from HF](https://github.com/huggingface/evaluate/tree/main/metrics/sacrebleu)
- EDNA: https://github.com/tingofurro/summac (SummaC-Conv)


### Helpfulness & Harmlessness (HH)

We use the Open LLM Leaderboard (v2) for evaluation on downstream NLP tasks, following the [official documentation](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about#reproducibility).
Scores are also normalized based on [this guide](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/normalization).

```bash
lm_eval --model_args="pretrained=${model},dtype=auto" \
  --tasks=leaderboard_ifeval \
  --batch_size=16 \
  --trust_remote_code \
  --apply_chat_template \
  --output_path="results_dir"

lm_eval --model_args="pretrained=${model},dtype=auto" \
  --tasks=leaderboard_bbh,leaderboard_gpqa,leaderboard_math_hard,leaderboard_mmlu_pro,leaderboard_musr \
  --batch_size=16 \
  --trust_remote_code \
  --output_path="results_dir"
```

We also perform evaluation on the [HumanRankEval benchmark](https://aclanthology.org/2024.naacl-long.456/) using the [official implementation](https://github.com/huawei-noah/noah-research/tree/master/NLP/HumanRankEval):

```bash
python main.py \
  --model auto_hf \
  --tasks human_rank_eval_* \
  --model_args pretrained="${model}" \
  --batch_size=32 \
  --data_path="huawei-noah/human_rank_eval" \
  --no_cache
```


### Text-to-Code Generation

We use the Bigcode evaluation harness framework from the [official repo]( https://github.com/bigcode-project/bigcode-evaluation-harness) to evaluate CodeLMs on HumanEval and MBPP datasets.

```bash
n_samples=100

for task in "humaneval" "mbpp"; do
    accelerate launch main.py \
    --model "${model}" \
    --tasks "${task}" \
    --max_length_generation 512 \
    --temperature 0.6 \
    --top_p 1.0 \
    --do_sample True \
    --n_samples "${n_samples}" \
    --batch_size "${n_samples}" \
    --precision fp16 \
    --save_generations \
    --generation_only \
    --save_generations_path "${out_dir}_n${n_samples}_t0.6.json"

    accelerate launch main.py \
    --model "${model}" \
    --tasks "${task}" \
    --temperature 0.6 \
    --top_p 1.0 \
    --n_samples "${n_samples}" \
    --allow_code_execution \
    --load_generations_path "${out_dir}_n${n_samples}_t0.6_${task}.json" \
    --metric_output_path "${out_dir}_n${n_samples}_t0.6_${task}_results.json"
```


## License

We follows Apache License Version 2.0. Please see the [License](./LICENSE) file for more information.

Disclaimer: This open source project is not an official Huawei product, Huawei is not expected to provide support for this project.



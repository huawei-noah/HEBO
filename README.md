# [DReSD: Dense Retrieval for Speculative Decoding](https://arxiv.org/pdf/2502.15572)

Resources for the paper (https://arxiv.org/pdf/2502.15572), contact ([Milan Gritta](mailto:milan.gritta@huawei.com)), if you have any queries.

### Abstract
Speculative decoding (SD) accelerates Large Language Model (LLM) generation by using an efficient draft model to propose 
the next few tokens, which are verified by the LLM in a single forward call, reducing latency while preserving its outputs. 
We focus on retrieval-based SD where the draft model retrieves the next tokens from a non-parametric datastore. 
Sparse retrieval ([REST](https://arxiv.org/pdf/2311.08252)), which operates on the surface form of strings, is currently the dominant paradigm due to 
its simplicity and scalability. However, its effectiveness is limited due to the usage of short contexts and exact string matching.
Instead, we introduce Dense Retrieval for Speculative Decoding (DReSD), a novel framework that uses approximate nearest neighbour 
search with contextualised token embeddings to retrieve the most semantically relevant token sequences for SD. Extensive experiments
show that DReSD achieves (on average) 87% higher acceptance rates, 65% longer accepted tokens and 19% faster generation speeds compared to sparse retrieval (REST).

### Installation

Inside your conda/venv environment, run ```pip install -r requirements.txt```

Inside the 'rest' folder is a **.whl** (backup) file to install 'draftretriever' (REST only) in case 
the pip install (draftretriever) fails for some reason. We used this **.whl** file to install the REST datastore.

### Datasets

For UltraChat, run the code below to create our sub-sampled corpus (optional, you can use the whole corpus, if you wish).
```
from datasets import load_dataset, Dataset
dataset = load_dataset("HuggingFaceH4/ultrachat_200k")['train_sft']
subset = []
for item in dataset.shuffle(42).select(range(80_000)):
    subset.append({
        'instruction': item['messages'][0]['content'], 
        'output': item['messages'][1]['content']
    })
new_dataset = Dataset.from_list(subset)
new_dataset.save_to_disk("/path/to/datasets/UltraChat")
```

### Preprocessing

1. Set the ```$PROJECT_ROOT``` to the folder where all DReSD code & outputs will be located.
2. Place the **dresd** code folder inside ```$PROJECT_ROOT```.
3. Inside ```$PROJECT_ROOT```, create a '**dresd_files**' folder for the DReSD outputs ```mkdir dresd_files```
4. Now ```cd dresd_files``` and create these folders: '**embeddings**', '**indices**', '**pca_models**' and '**random_samples**'. See the folder structure below.

```
$PROJECT_ROOT
    |-- dresd
    |-- dresd_files
        |-- embeddings
        |-- indices
        |-- pca_models
        |-- random_samples
```

5. Now you can go back to the DReSD code folder ```cd $PROJECT_ROOT/dresd/```

#### OOD Datastore

We first need to sample ~1M full-size embeddings to train the PCA model. The command below will save the embeddings
(file will have a suffix **_EMB**) into the 'random_samples' folder. These are the paper settings, feel free to go bigger!

```
deepspeed --include localhost:0 embed_data.py \
          --n_gpus 1 \
          --dataset EvolInstructCode \
          --prompt_key instruction \
          --model_name Llama2-Chat-7B \
          --max_length 1024 \
          --only_dump_big_emb True \
          --pca_dimension 64 \
          --path_prefix $PROJECT_ROOT
```

Next, run the PCA training using the following command. This will save the PCA model in the 'pca_models' folder.

```
python pca.py --dataset EvolInstructCode \
              --model_name Llama2-Chat-7B \
              --pca_dimension 64 \
              --path_prefix $PROJECT_ROOT
```

Run _embed_data.py_ again but with ```--only_dump_big_emb False``` and ```--n_len 20``` (how many next tokens to store, 
see "N" in the paper, we used 20, feel free to modify). Two files will be saved with suffixes **_META** and **_EMB**. 
This process will save the token embeddings from the entire dataset to the 'embeddings' folder, please be patient (takes a while).

```
deepspeed --include localhost:0 embed_data.py \
          --n_gpus 1 \
          --n_len 20 \
          --dataset EvolInstructCode \
          --prompt_key instruction \
          --model_name Llama2-Chat-7B \
          --max_length 1024 \
          --only_dump_big_emb False \
          --pca_dimension 64 \
          --path_prefix $PROJECT_ROOT
```

#### ID Datastore

In the paper, we showed that you can achieve much better SD results with the ID datastore. This step is quite slow because you need to generate
responses for all prompts auto-regressively. Run the command below, then **go back to the OOD Datastore steps (replace the dataset name!)**.
Every 2,500 steps/examples, the script will overwrite the latest version in case the process crashes for whatever reason. Feel free to modify!
By default, this will use greedy decoding, otherwise, scroll down and uncomment the nucleus hyperparameters (t=0.7, p=0.95).

```
deepspeed --include localhost:2,3 gen_data.py \
          --n_gpus 2 \
          --dataset EvolInstructCode \
          --prompt_key instruction \
          --model_name Llama2-Chat-7B \
          --max_tokens 128 \
          --path_prefix $PROJECT_ROOT
```

#### Building the Datastore(s)

Now you are ready to build the datastore(s). You can build a REST datastore using the following command.

```
python index.py --dataset EvolInstructCode \
                --datastore REST \
                --model_name Llama2-Chat-7B \
                --prompt_key instruction \
                --response_key output \
                --max_length 1024 \
                --path_prefix $PROJECT_ROOT 
```

You can build the DReSD datastore using the following command (takes a lot longer than REST, especially the intrinsic evaluation, 
which you can interrupt with CTRL+C if you're not interested in the MRR scores).

```
python index.py --dataset EvolInstructCode \
                --datastore DReSD \
                --model_name Llama2-Chat-7B \
                --prompt_key instruction \
                --response_key output \
                --max_length 1024 \
                --pca_dimension 64 \
                --path_prefix $PROJECT_ROOT 
```

### Running Speculative Decoding

You can now run (speculative) generation. The vanilla LLM (no SD, plain baseline) can be evaluated with the following command.
This will run inference with Llama2-Chat-7B for CodeAlpaca (code assistant) on 2 gpus (with DeepSpeed-TP).

```
deepspeed --include localhost:0,1 evaluate.py \
          --n_gpus 2 \
          --dataset CodeAlpaca \
          --prompt_key instruction \
          --model_name Llama2-Chat-7B \
          --max_tokens 128 \
          --max_length 2048 \
          --path_prefix $PROJECT_ROOT
```

Running DReSD on CodeAlpaca can be accomplished by using the following command. The draft shape can be controlled by ```num_drafts```
and ```len_drafts```, see paper for more details. 

```
deepspeed --include localhost:0,1 evaluate.py \
          --n_gpus 2 \
          --dresd True \
          --datastore EvolInstructCode \
          --num_drafts 10 \
          --len_drafts 10 \
          --dataset CodeAlpaca \
          --prompt_key instruction \
          --model_name Llama2-Chat-7B \
          --max_tokens 128 \
          --max_length 2048 \
          --pca_dimension 64 \
          --path_prefix $PROJECT_ROOT
```

REST has to be run on a **single gpu**! That's because each datastore call is **non-deterministic** hence it breaks 
in a multi-process run :/ Use the code below to run it on MT-Bench with 1 gpu :)

```
deepspeed --include localhost:0 evaluate.py \
          --n_gpus 1 \
          --rest True \
          --datastore UltraChat \
          --num_drafts 10 \
          --len_drafts 10 \
          --dataset MT-Bench \
          --prompt_key instruction \
          --model_name Llama2-Chat-7B \
          --max_tokens 128 \
          --max_length 2048 \
          --path_prefix $PROJECT_ROOT
```

Prompt Lookup Decoding (PLD) is one of the baselines we feature in the paper. You can run it with the following command:

```
deepspeed --include localhost:0,1 evaluate.py \
          --n_gpus 2 \
          --sd pld \
          --dataset CodeAlpaca \
          --prompt_key instruction \
          --len_drafts 10 \
          --model_name Llama2-Chat-7B \
          --max_tokens 128 \
          --max_length 2048 \
          --path_prefix $PROJECT_ROOT
```

Baseline SD with a small LLM is also featured in the paper, this is the 'classic' (original) speculative method. 
Use the following command to run SD with a LLama-Chat-68M model. PLD and 'vanilla' SD do not use ```--num_drafts```.

```
deepspeed --include localhost:0,1 evaluate.py \
          --n_gpus 2 \
          --sd small \
          --dataset CodeAlpaca \
          --prompt_key instruction \
          --len_drafts 10 \
          --model_name Llama2-Chat-7B \
          --small_name Llama-Chat-68M \
          --max_tokens 128 \
          --max_length 2048 \
          --path_prefix $PROJECT_ROOT
```

### Citing DReSD

```
@misc{gritta2025dresddenseretrievalspeculative,
      title={DReSD: Dense Retrieval for Speculative Decoding}, 
      author={Milan Gritta and Huiyin Xue and Gerasimos Lampouras},
      year={2025},
      eprint={2502.15572},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.15572}, 
}
```

### License

We follow Apache License Version 2.0. Please see the [License](./LICENSE) file for more information.

Disclaimer: This open source project is not an official Huawei product, Huawei is not expected to provide support for this project.

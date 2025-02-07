# [Code-Optimise: Optimising Code Language Models for Functional Correctness and Efficiency](https://arxiv.org/abs/2406.12502)

### Datasets
The train and validation sets of MBPP are used for Code-Optimise, while the test sets of MBPP and HumanEval are used for evaluation.

### Sampling
The `sample.sh` is used to sample solutions per problem to form the synthetic data. Example flags for running the script:

```
--regen True 
--data_path mbpp
--model_path StarCoder-1B
--n_seq: 100                # Outputs per prompt, defaults to 1.
--n_iter: 1                 # Iterations, defaults to 1. Generate n_seq // n_iter per iteration to save memory.
--temp: 0.6                 # Temperature, defaults to 1.0 if n_seq is 1.
--save_path sc-1b-ms-0.6
```

### Annotation
First, `sample.sh` is used to annotate the solutions by functional correctness and runtime. Example flags for running the script:

```
--regen False 
--data_path mbpp
--save_path sc-1b-ms-0.6
```

Then, `merge.sh` is used to merge and filter the annotated solutions to form the synthetic data. Example flags for running the script:

```
--data_path sc-1b-ms-0.6
--save_path sc-1b-ms-0.6
```

### Optimisation
The `train.sh` is used to sample solutions per problem to form the synthetic data. Example flags for running the script:

#### SFT
```
--data_path sc-1b-ms-0.6/mbpp
--model_path StarCoder-1B
--optim sft                         # Type of optimisation, either 'sft' or 'dpo'.
--top_p 25                          # Top-p of working solutions for SFT.
--augment True                      # Whether to use dynamic solution selection.
--save_path mbpp-1b-1b-sft-25
```

#### DPO
```
--data_path sc-1b-ms-0.6/mbpp
--model_path StarCoder-1B
--optim dpo
--task qvs                          # Type of preference pair for DPO: 'qvs', 'pvf', or 'all'.
--augment True
--save_path mbpp-1b-1b-dpo-qvs
```

### Evaluation
First, `eval.sh` is used to sample solutions per problem for evaluation. Example flags for running the script:

```
--regen True 
--data_path human_eval
--model_path mbpp-1b-1b-sft-25
--n_seq: 100
--n_iter: 1
--temp: 0.6
--save_path mbpp-1b-1b-sft-25-ms-0.6
```

Then, `eval.sh` is used to annotate the solutions by functional correctness and runtime. Example flags for running the script:

```
--regen False 
--data_path human_eval
--save_path mbpp-1b-1b-sft-25-ms-0.6
```

## Note
To ensure reliable runtimes, use the **same environment and machine** when timing the solutions. Furthermore, ensure **no unnecessary process is running** and **avoid using htop**.

## Cite

```
@article{gee2024code,
  title={Code-Optimise: Self-Generated Preference Data for Correctness and Efficiency},
  author={Gee, Leonidas and Gritta, Milan and Lampouras, Gerasimos and Iacobacci, Ignacio},
  journal={arXiv preprint arXiv:2406.12502},
  year={2024}
}
```

## License

We follow Apache license. Please see the [License](./LICENSE) file for more information.

Disclaimer: This open source project is not an official Huawei product, Huawei is not expected to provide support for this project.
# https://github.com/THUDM/LongBench/blob/main/pred.py
import os
from datasets import load_from_disk, load_dataset
import torch
import json
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf
from em_llm.utils import patch_hf, GreedySearch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import sys
import time
import numpy as np
from datetime import datetime
import pprint as pp


def print_dict(d, indent=0):
    for key, value in d.items():
        print('  ' * indent + str(key) + ':', end=' ')
        if isinstance(value, dict) or (str(value)[0] == '{' and str(value)[-1] == '}'):
            print()  # Move to the next line before printing the sub-dictionary
            print_dict(dict(value), indent + 1)
        else:
            print(value)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected. Got {v}, type: {type(v)}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--datetime", default=datetime.now().strftime('%Y-%m-%d %H:%M'))

    parser.add_argument("--allow_disk_offload", type=str2bool, default=False)

    args, extra_args = parser.parse_known_args()
    conf = OmegaConf.load(args.config_path)
    cli_conf = OmegaConf.from_cli(extra_args)
    conf = OmegaConf.merge(conf, cli_conf)    
    conf.output_dir_path = args.output_dir_path
    conf.rank = args.rank
    conf.world_size = args.world_size
    conf.datetime = args.datetime

    if torch.cuda.device_count() > 1:
        conf.model.use_hf_acc = True
    else:
        conf.model.use_hf_acc = False
    conf.model.allow_disk_offload = args.allow_disk_offload
    if args.allow_disk_offload:
        conf.model.world_size = args.world_size
    conf.model.disk_offload_dir = args.output_dir_path + f"/offload_data/{args.rank}"

    conf.model.tokenizer_path = conf.model.get("tokenizer_path", conf.model.path)
    conf.truncation = conf.get("truncation")

    datasets_str = args.datasets.strip().strip(",")
    datasets_list = datasets_str.split(",")
    conf.datasets = []
    for d in datasets_list:
        conf.datasets.append(d.strip())
    
    if args.rank is None or args.rank == 0:
        print_dict(dict(conf))
    return conf

def get_model_and_tokenizer(model_config, llm_device_map="cuda", rank=0):
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_path, trust_remote_code=True)
    attn_impl = model_config.get("attn_implementation", "sdpa")
    print(f"Using attention type: {attn_impl}")

    if model_config.use_hf_acc:
        print(f'Model split across {torch.cuda.device_count()} GPUs')
        import warnings
        with init_empty_weights() and warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = AutoModelForCausalLM.from_pretrained(model_config.path
                                                        , torch_dtype="auto"
                                                        , trust_remote_code=True
                                                        , attn_implementation=attn_impl)
        model = patch_hf(model, model_config.type, **model_config)
        model = load_checkpoint_and_dispatch(model, model_config.path
                                            , device_map="auto"
                                            , no_split_module_classes=["MistralDecoderLayer", "LlamaDecoderLayer", "Phi3DecoderLayer", "ContextManager"])   
        print(f'Worker {rank}: spreading model on {torch.cuda.device_count()} GPUs with device map: ', model.hf_device_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_config.path
                                                     , torch_dtype="auto"
                                                     , trust_remote_code=True
                                                     , device_map=llm_device_map
                                                     , attn_implementation=attn_impl)
        model = patch_hf(model, model_config.type, **model_config)
    
    return model, tokenizer

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, conv_type, max_length, args):
    conv_type = conv_type.strip().lower()
    if conv_type == "vicuna":
        from fastchat.conversation import get_conv_template
        conv = get_conv_template("vicuna_v1.1")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif conv_type in ["mistral-inst", "qwen", "minicpm", "llama-3-inst", "phi-3-mini-inst"]:
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        raise NotImplementedError

    return prompt

def extend_passkey_context(context, passkey, max_len=512, tokenizer=None):
    # Given variables
    noise = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    passkey_sent = "The pass key is {}. Remember it. {} is the pass key."

    passkey_sentence = passkey_sent.format(passkey,passkey)
    if tokenizer is None:
        context_len = len(context.split())
        noise_len = len(noise.split())
        desired_len = int(max_len * 1000 * 0.79) # desired context length in terms of 1000s of tokens, with word to token ratio of 0.79 (for Mistral-7B)
    else:
        context_len = len(tokenizer(context))
        noise_len = len(tokenizer(noise))
        desired_len = max_len * 1000 # desired context length in terms of 1000s of tokens
    remaining_len = desired_len - context_len
    insertions = remaining_len // noise_len

    passkey_index = context.index(passkey_sentence)
    passkey_sentence += " "

    passkey_pos = passkey_index / len(context)
    pre_insertions = int(passkey_pos * insertions)
    post_insertions = insertions - pre_insertions

    pre_noise = noise * pre_insertions
    post_noise = noise * post_insertions

    new_context = context[:passkey_index] + pre_noise + passkey_sentence + \
                   post_noise + context[passkey_index + len(passkey_sentence):]
    if tokenizer is None:
        assert desired_len - noise_len <= len(new_context.split()) <= desired_len + noise_len, f"New context length: {len(new_context.split())} does not match the desired length: {desired_len}"
    else:
        new_context_len = len(tokenizer(new_context))
        assert desired_len - noise_len <= new_context_len <= desired_len + noise_len, f"New context length: {new_context_len} does not match the desired length: {desired_len}"
    assert str(passkey) in new_context, "Passkey not found in the new context."
    return new_context    

def load_infinite_bench(path, data_name, **kwargs) -> str:
    import re
    """
    Create prompt for a given example.

    Args:
        eg: example dict
        data_name: name of the dataset/task
    """
    print(f"read {data_name}.jsonl")
    fin = open(os.path.join(path, data_name.replace("__long", "") + ".jsonl"), "r")
    lines = fin.readlines()
    fin.close()
    data = [json.loads(line) for line in lines]
    def get_answer(inp: dict):
        if data_name in ["code_debug", "longbook_choice_eng"]:
            OPTIONS = "ABCD"
            if isinstance(inp["answer"], str):
                ret = [inp["answer"], OPTIONS[inp['options'].index(inp["answer"])]]
            elif isinstance(inp["answer"], list):
                if len(inp["answer"]) == 1:
                    ret = [inp["answer"][0], OPTIONS[inp['options'].index(inp["answer"][0])]]
                elif len(inp["answer"]) == 2 and inp["answer"][1] in ['A', 'B', 'C', 'D']:
                    ret = inp['answer']
                else:
                    raise ValueError
            else:
                raise ValueError
            return ret
        return inp["answer"]

    ret = []
    for eg in data:
        # ================= Code tasks
        if data_name == "code_run":
            find_result = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", eg['input'])
            func_call = find_result[0]
            func = func_call.split("(")[0]
            instance = {"func": func, "func_call": func_call, "context": eg["context"]}
        elif data_name in ["code_debug", "code_debug_qa"]:
            # Load source code
            instance = {"context": eg["context"]}
            if data_name == "code_debug":
                instance.update({
                    "OPTION_A": eg["options"][0],
                    "OPTION_B": eg["options"][1],
                    "OPTION_C": eg["options"][2],
                    "OPTION_D": eg["options"][3]})
        # ================= Code tasks
        elif data_name == "longdialogue_qa_eng":
            instance = {"context": eg["context"]}
        # ==================== Long book tasks
        elif data_name in [
            "longbook_choice_eng",
            "longbook_qa_eng",
            "longbook_sum_eng",
            "longbook_qa_chn",
        ]:
            instance = {"context": eg["context"]}
            if data_name == "longbook_choice_eng":
                instance.update({
                    "question": eg["input"],
                    "OPTION_A": eg["options"][0],
                    "OPTION_B": eg["options"][1],
                    "OPTION_C": eg["options"][2],
                    "OPTION_D": eg["options"][3],
                })
            elif data_name in ["longbook_qa_eng", "longbook_qa_chn"]:
                instance.update({
                    "question": eg["input"],
                })
        elif data_name == "math_calc":
            instance = {"context": eg["context"]}
        elif data_name == "math_find":
            prompt = eg['input']
            context = eg['context']
            # Find "the * number" from the prompt
            find_result = re.findall(r"The .+ of", prompt)
            assert find_result, f"Cannot find the target number in {prompt}"
            target_number = find_result[0].lower()[:-3]
            # Replace the number with the answer
            prefix = f"What is {target_number} in the following list?"
            instance = {"prefix": prefix, "context": context, "input": prompt}
        elif data_name == "kv_retrieval":
            instance = {
                "context": eg["content"] if "content" in eg else eg["context"],
                "input": eg["input"],
                "key": eg["input"][6:44]
            }
            assert eg['input'][6] == '"'
            assert eg['input'][43] == '"'
        elif data_name == "passkey__long":
            instance = {
                "context": extend_passkey_context(eg["context"]
                                                    , eg['answer'][0]
                                                    , max_len=kwargs.get("extended_passkey", 1024)),
                "input": eg["input"],
            }
        else:
            instance = {
                "context": eg["content"] if "content" in eg else eg["context"],
                "input": eg["input"],
            }
        ans = get_answer(eg)
        instance["answers"] = ans if isinstance(ans, list) else [ans]
        instance["length"] = len(instance["context"].split())
        instance["all_classes"] = None

        ret.append(instance)
        # if len(ret) > 4:
        #     break
    return ret

def load_data(dataset: str, **kwargs):
    if dataset.replace("__long", "") in set([
        "kv_retrieval", 
        "passkey", 
        "number_string", 
        "code_run", 
        "code_debug", 
        "longdialogue_qa_eng", 
        "longbook_qa_eng", 
        "longbook_sum_eng", 
        "longbook_choice_eng", 
        "longbook_qa_chn", 
        "math_find", 
        "math_calc"
    ]):
        path = "benchmark/data/infinite-bench"
        data = load_infinite_bench(path, dataset, **kwargs)

    elif "pg19" in dataset:
        if 'train' in dataset:
            split = 'train'
        elif 'val' in dataset:
            split = 'validation'
        elif 'test' in dataset:
            split = 'test'
        else:
            split = ''
        path = f"benchmark/data/pg19/{split}"
        data = load_from_disk(path)
        if split != '':
            dataset = f"pg19-{split}"
    else:
        try:
            data = load_from_disk(
                f"benchmark/data/longbench/{dataset}"
            )
        except:
            data = load_dataset('THUDM/LongBench', dataset, split='test', trust_remote_code=True)
            data.save_to_disk(os.path.join(f"benchmark/data/longbench", f"{dataset}"))

    print(f"Pred {dataset}")
    return data, dataset

def get_past_ids(out_path):
    past_ids = []
    if os.path.exists(out_path):
        with open(out_path, 'r') as file:
            for line in file:
                entry = json.loads(line)
                past_ids.append(entry['id'])
    return past_ids

def post_process(pred, conv_type, dataset):
    if conv_type == "qwen":
        pred = pred.split("<|im_end|>")[0]
    if "phi" in conv_type:
        pred = pred.strip()
    if "llama" in conv_type.lower():
        pred = pred.split("<|eot_id|>")[0]
    elif dataset == "samsum":
        pred = pred.split("\n")[0].strip()
    return pred

def get_pred(
    searcher,
    tokenizer, 
    model_type,
    data, 
    max_length: int,
    max_gen: int, 
    prompt_format: str, 
    dataset, 
    conv_type: str, 
    gen_chunk_size = None, 
    truncation: str = None, 
    rank: int = None, 
    world_size: int = None,
    verbose: bool = False,
    out_path: str = None,
    return_block_size = False,
    args = None,
    past_ids = [],
):
    def get_id(cur):
        if world_size is None:
            return cur
        else:
            return world_size * cur + rank
        
    preds = []
    data = list(data)

    if world_size is not None:
        data = data[rank::world_size]

    cur = 0
    total = len(data)

    if args.em_splitter == 'sentence':
        print("loading EN spacy")
        import spacy
        nlp = spacy.load('en_core_web_sm')
        nlp.max_length = 2147483647

    sent_len = []
    for json_obj in tqdm(data):

        cur_id = get_id(cur)

        long = len(json_obj['context'].split()) > args.model.disk_offload_threshold
        too_long = long and not (torch.cuda.device_count() >= 3 or (torch.cuda.device_count() >= 2 and args.model.allow_disk_offload) \
                    or (torch.cuda.device_count() == 1 and args.model.vector_offload))
        if too_long:
            print(f"Skipping {cur_id} due to length: {len(json_obj['context'].split())}")
        
        if cur_id not in past_ids and not too_long:

            prompt = prompt_format.format(**json_obj)

            extra_end_token_ids = []
            if conv_type == "llama-3-inst":
                extra_end_token_ids.append(tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0])
            elif conv_type == "phi-3-mini-inst":
                extra_end_token_ids.append(tokenizer.encode("<|end|>", add_special_tokens=False)[0])
            elif conv_type == "qwen":
                extra_end_token_ids.append(tokenizer.encode("<|im_end|>", add_special_tokens=False)[0])

            if dataset == "samsum":
                extra_end_token_ids.append(tokenizer.encode("\n", add_special_tokens=False)[-1])

            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: 
                # chat models are better off without build prompts on these tasks
                prompt = build_chat(tokenizer, prompt, conv_type, max_length, args)

                if conv_type.strip().lower() in ['mistral-inst']:
                    add_special_tokens = False
                else:
                    add_special_tokens = True
            else:
                add_special_tokens = True

            if args.em_splitter == 'sentence':
                tokenized_prompt = []
                em_labels = []
                sentences = nlp(prompt).sents
                for sent in sentences:
                    sent_inp_ids = tokenizer(sent.text, truncation=False, return_tensors="pt",
                                            add_special_tokens=add_special_tokens).input_ids[0]
                    sent_len.append(len(sent_inp_ids))

                    tokenized_prompt.append(sent_inp_ids)
                    boundaries = torch.zeros(len(sent_inp_ids))
                    boundaries[0] = 1
                    em_labels.append(boundaries)
                tokenized_prompt = torch.hstack(tokenized_prompt)
                em_labels = torch.hstack(em_labels).bool()
            else:
                tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]
                em_labels = None

            if truncation is None:
                if len(tokenized_prompt) > max_length - max_gen:
                    if verbose:
                        print(f"Length {len(tokenized_prompt)}. Skipped.")
                    continue

            else:
                if truncation == "suffix":
                    length = len(tokenized_prompt)
                    if length > max_length - max_gen:
                        if verbose:
                            print("over length")
                        init_token_num = 128
                        prompt = tokenizer.decode(tokenized_prompt[:init_token_num].tolist() + tokenized_prompt[- (max_length - max_gen - init_token_num):].tolist())
                        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]
                else:
                    raise NotImplementedError

            offloading_args = {
                "vector_offload_threshold": args.model.vector_offload_threshold,
                "disk_offload_threshold": args.model.disk_offload_threshold
            }
            time1 = time.time()
            output = searcher.generate(
                input_ids=tokenized_prompt,
                em_labels=em_labels,
                max_length=max_gen,
                chunk_size=gen_chunk_size,
                extra_end_token_ids=extra_end_token_ids,
                **offloading_args
            )
            time2 = time.time()

            pred = post_process(output["pred"], conv_type, dataset)
            if model_type == "em-llm" and return_block_size:
                block_sizes = [block.size for block in searcher.past_kv[0].global_blocks[0]]
                mean_block_size = np.mean(np.array(block_sizes))
            else:
                block_sizes = None
                mean_block_size = None

            preds.append(
                {
                    "id": cur_id, 
                    "pred": pred, 
                    "answers": json_obj.get("answers"), 
                    "all_classes": json_obj.get("all_classes"), 
                    "length": json_obj.get("length"), 
                    "token_length": len(tokenized_prompt) + max_gen, 
                    "chunk_ppl": output.get("chunk_ppl"), 
                    "total_ppl": output.get("total_ppl"),
                    "block_sizes": block_sizes, 
                    "mean_block_size": mean_block_size, 
                    "generation_time": time2 - time1
                }
            )

            if verbose:
                print(f"----------{cur}/{total}----------")
                print("Length: ", len(tokenized_prompt))
                print("Question:", prompt[-100:])
                print("Pred:", pred)
                print("Answer:", json_obj.get("answers"))
                print("")

            with open(out_path, "a+", encoding="utf-8") as f:
                json.dump(preds[-1], f, ensure_ascii=False)
                f.write('\n')

        searcher.clear()
        cur += 1

    if args.em_splitter == 'sentence':
        print(f"Avg sentence length: {sum(sent_len) / len(sent_len)}")
    return preds

class DualLogger:
    def __init__(self, filename, mode='a', rank=0, world_size=1):
        self.terminal = sys.stdout
        self.log = open(filename, mode, buffering=1, encoding='utf-8')  # Line-buffering mode.
        self.rank = rank
        self.world_size = world_size

    def write(self, message):
        worker_message = f"Worker{self.rank}: " + str(message) if self.world_size > 1 else str(message) 
        self.terminal.write(worker_message)
        self.log.write(message)

    def flush(self):  # Needed for compatibility with flush operations
        self.terminal.flush()
        self.log.flush()

def main(args):

    output_dir_path = args.output_dir_path
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    if args.logging:
        log_dir_path = os.path.join(output_dir_path, f"logs/{args.datetime}/")
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)
        log_path = f'worker_{args.rank}.log'
        sys.stdout = sys.stderr = DualLogger(
            os.path.join(log_dir_path, log_path), 
            rank=args.rank, 
            world_size=args.world_size
        )

    if not args.model.use_hf_acc and args.model.allow_disk_offload:
        args.model.vector_offload = True
    else: 
        args.model.vector_offload = False
    model, tokenizer = get_model_and_tokenizer(args.model, rank=args.rank)
    searcher = GreedySearch(
        model, 
        tokenizer, 
        args.model.type, 
        em_splitter=args.em_splitter, 
        compute_ppl=args.compute_ppl,
    )

    datasets = args.datasets
    
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("benchmark/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("benchmark/config/dataset2maxlen.json", "r"))

    multiprocessing = args.world_size is not None and args.world_size > 1
    if multiprocessing:
        assert args.rank in list(range(args.world_size))

    # predict on each dataset
    for dataset in datasets:
        data, dataset = load_data(dataset, extended_passkey=args.get("extended_passkey", 1024))
        prompt_format = dataset2prompt[dataset.replace("__long", "")]
        max_gen = dataset2maxlen[dataset.replace("__long", "")]

        out_path = os.path.join(output_dir_path, f"{dataset}.jsonl")
        past_ids = get_past_ids(out_path)

        get_pred(
            searcher=searcher,
            tokenizer=tokenizer,
            model_type=args.model.type,
            data=data,
            max_length=args.max_len,
            max_gen=max_gen,
            prompt_format=prompt_format,
            dataset=dataset,
            conv_type=args.conv_type,
            gen_chunk_size=args.chunk_size,
            truncation=args.truncation,
            rank=args.rank,
            world_size=args.world_size,
            verbose=args.verbose,
            out_path=out_path,
            return_block_size=args.return_block_size,
            args=args,
            past_ids=past_ids,
        )


if __name__ == '__main__':

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args = parse_args()

    main(args)
    



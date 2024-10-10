# https://github.com/THUDM/LongBench/blob/main/eval.py
import os
import json
import argparse
import numpy as np

def parse_json_stream(stream):
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(stream):
        try:
            obj, idx_new = decoder.raw_decode(stream, idx)
            yield obj
            idx = idx_new
        except json.JSONDecodeError:
            # Move to the next character and try again
            raise ValueError

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

from infinitebench_eval import (
    get_score_one_kv_retrieval,
    get_score_one_kv_retrieval,
    get_score_one_kv_retrieval,
    get_score_one_passkey,
    get_score_one_number_string,
    get_score_one_code_run,
    get_score_one_code_debug,
    get_score_one_longdialogue_qa_eng,
    get_score_one_longbook_qa_eng,
    get_score_one_longbook_sum_eng,
    get_score_one_longbook_choice_eng,
    get_score_one_longbook_qa_chn,
    get_score_one_math_find,
    get_score_one_math_calc,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
    "narrativeqa-full": qa_f1_score,
    
    # Retrieve
    "kv_retrieval": get_score_one_kv_retrieval,
    "kv_retrieval_prefix": get_score_one_kv_retrieval,
    "kv_retrieval_both": get_score_one_kv_retrieval,

    "passkey": get_score_one_passkey,
    "number_string": get_score_one_number_string,
    # Code
    "code_run": get_score_one_code_run,
    "code_debug": get_score_one_code_debug,
    # Longbook
    "longdialogue_qa_eng": get_score_one_longdialogue_qa_eng,
    "longbook_qa_eng": get_score_one_longbook_qa_eng,
    "longbook_sum_eng": get_score_one_longbook_sum_eng,
    "longbook_choice_eng": get_score_one_longbook_choice_eng,
    "longbook_qa_chn": get_score_one_longbook_qa_chn,
    # Math
    "math_find": get_score_one_math_find,
    "math_calc": get_score_one_math_calc,
    #pg19
    "pg19": lambda *args: None,
}

def flatten_list(input_list):
    if not isinstance(input_list, list):  # Handles the case when the input is not a list
        return [input_list]
    
    flat_list = []
    for item in input_list:
        if isinstance(item, list):
            # Recursively flatten the item if it is a list
            flat_list.extend(flatten_list(item))
        else:
            # Add the item directly if it is not a list
            flat_list.append(item)
    return flat_list

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores


def calc_score(dataset, prediction, ground_truths, all_classes):
    if dataset in ["code_debug"]:
        return get_score_one_code_debug(prediction, ground_truths)
    score = 0.
    for ground_truth in ground_truths:
        score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
    return score


def scorer(dataset, predictions, answers, all_classes):
    if 'pg19' in dataset:
        return None, len(predictions)
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        score = calc_score(dataset, prediction, ground_truths, all_classes)
        total_score += score
    print(f"{dataset} - Score: {total_score}/{len(predictions)} -> {round(100 * total_score / len(predictions), 2)}")
    return round(100 * total_score / len(predictions), 2), len(predictions)

if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    path = args.dir_path
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        try:
            if not filename.endswith("jsonl"):
                continue
            predictions, answers, lengths, ppl, block_sizes = [], [], [], [], []
            ids = []
            dataset = filename.split('.')[0].replace("__long","")
            with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    data = json.loads(line)
                    if data['id'] in ids:
                        continue
                    else:
                        ids.append(data['id'])

                    predictions.append(data["pred"])
                    ppl.append(data.get("total_ppl"))
                    if data.get('block_sizes') is not None:
                        block_sizes.extend(data.get('block_sizes'))
                    answers.append(data.get("answers"))
                    all_classes = data.get("all_classes")

                    if "length" in data:
                        lengths.append(data["length"])

            if dataset.endswith("_e"):
                _dataset = dataset.rstrip("_e")
            else:
                _dataset = dataset
            if _dataset[:8] == "longeval":
                _dataset = "longeval"
            if args.e:
                score = scorer_e(_dataset, predictions, answers, lengths, all_classes)
            else:
                score, len_predictions = scorer(_dataset, predictions, answers, all_classes)
            ppl = np.array(ppl)
            block_sizes = np.array(block_sizes)
            results = {
                "score": score, 
                "len_predictions": len_predictions
            }
            if not (any(p is None for p in ppl)):
                results["ppl_mean"] = np.mean(ppl)
                results["ppl_std"] = np.std(ppl)
            if len(block_sizes) > 0:
                results["block_sizes_mean"] = np.mean(block_sizes)
                results["block_sizes_std"] = np.std(block_sizes) 
            scores[dataset] = results
        except Exception as err:
            print(f'Error processing line {i} of file {filename}: ', err)
            raise err

    out_path = os.path.join(args.dir_path, "result.json")
    print(scores)
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

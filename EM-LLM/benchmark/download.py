from datasets import load_dataset
import os

all_datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

for dataset in all_datasets:
    data = load_dataset('THUDM/LongBench', dataset, split='test', trust_remote_code=True)
    data.save_to_disk(os.path.join(f"benchmark/data/longbench", f"{dataset}"))

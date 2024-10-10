[ ! -d "benchmark/data" ] && mkdir -p benchmark/data
[ ! -d "benchmark/data/infinite-bench" ] && mkdir benchmark/data/infinite-bench
[ ! -d "benchmark/data/longbench" ] && mkdir benchmark/data/longbench

python benchmark/download.py

cd benchmark/data/infinite-bench

files=(
    "math_find.jsonl"
    "number_string.jsonl"
    "passkey.jsonl"
    "longbook_choice_eng.jsonl"
    "kv_retrieval.jsonl"
    "code_debug.jsonl"
    "longbook_qa_eng.jsonl"
    "longbook_sum_eng.jsonl"
)

# Base URL for downloads
base_url="https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/main"

# Download files if they don't already exist
for file in "${files[@]}"; do
    if [ ! -f "$file" ]; then
        wget "${base_url}/${file}" --no-check-certificate
    fi
done
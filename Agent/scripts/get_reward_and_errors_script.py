import pandas as pd

run_path = "exp_llm@_agents.0.llm=llama,method=direct,task=babyai/2023-11-13_10-45-44/"

file_path = f"logs/runs/{run_path}/output.jsonl"

data = pd.read_json(path_or_buf=file_path, lines=True)

# check number of occurrences of reward="fail"
fail_count = data[data["reward"] == "fail"].shape[0]
print(f"Fails: {fail_count}")

# log dict counting occurrences of error messages
log = {}
errors = data[data["error"].notna()]["error"] if "error" in data.columns else []
assert len(errors) == fail_count

for error in errors:
    # split error on first set of square brackets
    error = error.split("\n")[-2].split("[", 1)[0]
    if error in log:
        log[error] += 1
    else:
        log[error] = 1
print(f"Error counts: {log}")

# remove all instances of reward="fail"
data = data[data["reward"] != "fail"]

# calculate average reward
num_episodes = data["episode"].nunique()
print(f"Reward avg: {data['reward'].sum() / num_episodes}")

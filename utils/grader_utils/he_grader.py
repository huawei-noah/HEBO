from utils.grader_utils.he_check import evaluate_functional_correctness

def entry_point(
    sample_file: str,
    k: str = "1,2,3,4,5,6,7,8",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = None,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    print(results)

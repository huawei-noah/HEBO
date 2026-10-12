import argparse
import yaml
import json
import os
import random
import numpy as np
import torch
from tqdm import tqdm

from utils.data_utils import load_data, process_item

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class PowerDistributionSampler:
    def __init__(
        self, model_name: str, gpu_memory_utilization: float = 0.9,
        seed: int = 0, vllm_engine: str = "V1",
    ):
        if vllm_engine != "V1":
            raise ValueError("Power sampling requires V1")
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        from importlib.metadata import version
        if version("vllm") != "0.8.3":
            raise RuntimeError("Power sampling requires vllm==0.8.3")
        from vllm import LLM, SamplingParams

        self.sampling_params = SamplingParams
        print(f"Loading model: {model_name} (engine={vllm_engine}, seed={seed})")
        self.llm = LLM(
            model=model_name,
            dtype="auto",
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            seed=seed,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def _generate(self, **kwargs):
        return self.llm.generate(**kwargs)

    def _compute_jackknife_estimator(self, candidate_logs, rollout_logs, m_t):
        """Combine joint log weights with leave-one-out bias correction."""
        candidate_logs = np.asarray(candidate_logs, dtype=np.float64)
        rollout_logs = np.asarray(rollout_logs, dtype=np.float64)
        if m_t < 2 or rollout_logs.shape != (len(candidate_logs), m_t):
            raise ValueError("Invalid rollout dimensions")
        if not np.isfinite(candidate_logs).all() or not np.isfinite(rollout_logs).all():
            raise ValueError("Non-finite sampling log weights")

        def distribution(logs):
            log_zeta = np.logaddexp.reduce(logs, axis=1) - np.log(logs.shape[1])
            weights = candidate_logs + log_zeta
            return np.exp(weights - np.logaddexp.reduce(weights))

        full = distribution(rollout_logs)
        leave_one_out = np.zeros_like(full)
        for index in range(m_t):
            leave_one_out += distribution(np.delete(rollout_logs, index, axis=1))
        corrected = m_t * full - ((m_t - 1) / m_t) * leave_one_out
        corrected = np.maximum(corrected, 0.0)
        return full if corrected.sum() == 0 else corrected / corrected.sum()

    def generate_trajectory(
        self,
        prompt: str,
        max_len: int,
        alpha: float,
        k_t: int,
        m_t: int,
        temperature: float,
        block_size: int = 192,
        lookahead_size: int = 192,
        stop_on_answer: bool = True,
        l_t: int | None = None,
    ) -> str:
        l_t = k_t if l_t is None else l_t
        if not 1 <= k_t <= l_t:
            raise ValueError("Require l_t >= k_t >= 1")
        prompt_token_ids = self.tokenizer.encode(prompt)

        CONTEXT_LIMIT = 4096
        SAFE_INPUT_LIMIT = CONTEXT_LIMIT - 100

        if len(prompt_token_ids) > SAFE_INPUT_LIMIT:
            prompt_token_ids = prompt_token_ids[-SAFE_INPUT_LIMIT:]

        current_token_ids = list(prompt_token_ids)
        prompt_len = len(prompt_token_ids)

        max_total_tokens = min(prompt_len + max_len, CONTEXT_LIMIT)
        eos_token_id = self.tokenizer.eos_token_id

        while len(current_token_ids) < max_total_tokens:
            cand_params = self.sampling_params(
                n=l_t,
                temperature=temperature,
                max_tokens=min(block_size, max_total_tokens - len(current_token_ids)),
                seed=None,
                logprobs=1,
                top_p=0.99
            )
            cand_outputs = self._generate(
                prompt_token_ids=[current_token_ids],
                sampling_params=cand_params,
                use_tqdm=False
            )
            if len(cand_outputs) != 1 or len(cand_outputs[0].outputs) != l_t:
                raise ValueError("Missing candidate samples")
            candidates = cand_outputs[0].outputs
            for cand in candidates:
                if cand.cumulative_logprob is None or not np.isfinite(cand.cumulative_logprob):
                    raise ValueError("Missing or non-finite candidate logp")
            if l_t > k_t:
                candidates = sorted(
                    candidates, key=lambda cand: cand.cumulative_logprob, reverse=True
                )[:k_t]
            candidate_logweights = np.asarray(
                [alpha * cand.cumulative_logprob for cand in candidates], dtype=np.float64
            )

            active_indices = [
                i for i, cand in enumerate(candidates)
                if not (stop_on_answer and eos_token_id in cand.token_ids)
            ]
            lookahead_prompts = [
                current_token_ids + list(candidates[i].token_ids)
                for i in active_indices
            ]
            rem_after_block = (
                max_total_tokens - len(lookahead_prompts[0]) if active_indices else 0
            )
            actual_lookahead = min(lookahead_size, rem_after_block)

            if not active_indices or actual_lookahead <= 5:
                probs = np.exp(candidate_logweights - np.logaddexp.reduce(candidate_logweights))
                selected_idx = np.random.choice(len(candidates), p=probs)
                current_token_ids.extend(candidates[selected_idx].token_ids)
                break

            look_params = self.sampling_params(
                n=m_t,
                temperature=temperature,
                max_tokens=actual_lookahead,
                min_tokens=0,
                seed=None,
                logprobs=1,
                top_p=1.0,
                top_k=-1
            )
            outputs = self._generate(
                prompt_token_ids=lookahead_prompts,
                sampling_params=look_params,
                use_tqdm=False
            )

            # Empty continuations of completed candidates have log weight zero.
            rollout_logweights = np.zeros((len(candidates), m_t), dtype=np.float64)
            if len(outputs) != len(active_indices):
                raise ValueError("Missing rollout request outputs")
            for k_idx, request_output in zip(active_indices, outputs):
                if len(request_output.outputs) != m_t:
                    raise ValueError("Missing rollout samples")
                for m_idx, completion in enumerate(request_output.outputs):
                    logp = completion.cumulative_logprob
                    if logp is None:
                        raise ValueError("Missing rollout logp")
                    rollout_logweights[k_idx, m_idx] = (alpha - 1) * logp

            q_dist = self._compute_jackknife_estimator(
                candidate_logweights, rollout_logweights, m_t)
            selected_idx = np.random.choice(len(candidates), p=q_dist)

            chosen_block = candidates[selected_idx].token_ids
            current_token_ids.extend(chosen_block)

            if stop_on_answer and eos_token_id in chosen_block:
                break

        return self.tokenizer.decode(current_token_ids[prompt_len:])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--seed", type=int, help="Override the configured seed")
    parser.add_argument("--output", help="Override the output JSONL path")
    parser.add_argument("--limit", type=int, help="Maximum number of problems")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.output is not None:
        cfg["output_file"] = args.output

    model_name = cfg["model_name"]
    model_alias = cfg.get("model_alias", "qwen")
    dataset_name = cfg["dataset"]["name"]
    data_path = cfg["dataset"].get("path")
    split = cfg["dataset"].get("split", "test")

    alpha = float(cfg.get("alpha", 4.0))
    k_t = int(cfg.get("k_t", 8))
    l_t = int(cfg.get("l_t", k_t))
    m_t = int(cfg.get("m_t", 8))
    temperature = float(cfg.get("temperature", 0.25))
    max_tokens = int(cfg.get("max_new_tokens", 3072))

    block_size = int(cfg.get("block_size", 192))
    lookahead_size = int(cfg.get("lookahead_size", 192))
    if block_size < 1 or lookahead_size < 0 or not 1 <= k_t <= l_t or m_t < 2:
        raise ValueError("Require block_size >= 1, lookahead_size >= 0, l_t >= k_t >= 1, m_t >= 2")
    base_seed = int(cfg.get("seed", 0))
    output_file = cfg["output_file"]

    vllm_engine = cfg.get("vllm_engine", "V1")
    sampler = PowerDistributionSampler(model_name, seed=base_seed, vllm_engine=vllm_engine)

    print(f"Loading {dataset_name}...")
    dataset = load_data(dataset_name, split, data_path)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    set_seed(base_seed)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        dataset = list(dataset)[:args.limit]
    print(f"Starting generation with Temp={temperature}...")

    with open(output_file, "x", encoding="utf-8") as f:
        for i, item in tqdm(enumerate(dataset), total=len(dataset)):
            input_text, solution, metadata = process_item(
                item,
                dataset_name,
                model_alias,
                sampler.tokenizer,
            )

            completion = sampler.generate_trajectory(
                prompt=input_text,
                max_len=max_tokens,
                alpha=alpha,
                k_t=k_t,
                l_t=l_t,
                m_t=m_t,
                block_size=block_size,
                lookahead_size=lookahead_size,
                temperature=temperature,
            )

            raw_id = metadata.get("task_id") or metadata.get("id")
            if raw_id:
                task_id = raw_id
            else:
                task_id = str(i)

            out_obj = {
                "task_id": task_id,
                "prompt": input_text,
                "completion": completion,
                "solution": solution,
                "config": {
                    "model_name": model_name,
                    "model_alias": model_alias,
                    "dataset": cfg["dataset"],
                    "block_size": block_size,
                    "lookahead_size": lookahead_size,
                    "max_new_tokens": max_tokens,
                    "alpha": alpha,
                    "k_t": k_t,
                    "l_t": l_t,
                    "m_t": m_t,
                    "temperature": temperature,
                    "seed": base_seed,
                    "scoring": "raw_joint_logprob",
                    "rollout_log_weight": "(alpha-1)*logp",
                    "candidate_top_p": 0.99,
                    "rollout_top_p": 1.0,
                    "rollout_min_tokens": 0,
                    "vllm_backend": "0.8.3-native",
                    "eos_handling": "terminal",
                    "vllm_engine": vllm_engine
                }
            }
            f.write(json.dumps(out_obj) + "\n")
            f.flush()

    print(f"Done. Results saved to {output_file}")

if __name__ == "__main__":
    main()

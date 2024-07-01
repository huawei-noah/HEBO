import multiprocessing
import os
import time

import hydra
import numpy as np
import pyrootutils
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from agent import utils
from agent.models.llm import BatchedLLM
from agent.train.env_worker import EnvWorker
from agent.utils import pylogger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = pylogger.get_pylogger(__name__)

# Tokenizer are already parallelized through env workers
os.environ["TOKENIZERS_PARALLELISM"] = "False"


@hydra.main(version_base="1.3", config_path="../../configs", config_name="default_sa_train_rlft")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)
    OmegaConf.resolve(cfg)
    del cfg._agents

    seed_gpu_wise = 0
    local_rank = 0
    if os.environ.get("LOCAL_RANK") is not None:
        # distributed setup, accelerator object will be build inside PPOTrainer
        local_rank = int(os.environ.get("LOCAL_RANK"))
        world_size = int(os.environ.get("WORLD_SIZE"))
        torch.cuda.set_device(local_rank)
        cfg.agents[0].llm.model_kwargs.device_map = f"cuda:{local_rank}"
        seed_gpu_wise += local_rank

        # split buffer_size across world size
        assert cfg.training.config.buffer_size % world_size == 0, "buffer_size must be divisible per world_size"
        cfg.training.config.buffer_size = cfg.training.config.buffer_size // world_size

    cfg.max_episodes = cfg.training.config.buffer_size // cfg.training.config.env_workers

    # check and set max_episodes, max_env_steps and max_thinking_steps
    assert (
        isinstance(cfg.max_episodes, int) and cfg.max_episodes > 0
    ) or cfg.max_episodes is None, "max_episodes should be a positive integer or None."
    assert (
        isinstance(cfg.max_env_steps, int) and cfg.max_env_steps > 0
    ) or cfg.max_env_steps is None, "max_env_steps should be a positive integer or None."
    assert (
        isinstance(cfg.max_thinking_steps, int) and cfg.max_thinking_steps >= 0
    ) or cfg.max_thinking_steps is None, "max_thinking_steps should be a positive integer or None."

    logger = hydra.utils.instantiate(cfg.logger, project_cfg=cfg, _recursive_=False)
    llms = [hydra.utils.instantiate(c.llm) for c in cfg.agents]
    seed = 42 if llms[0].seed is None else llms[0].seed
    llm_support_async_calls = all(llm.support_async_call for llm in llms) or cfg.training.config.env_workers == 1
    if not llm_support_async_calls:
        llms = [BatchedLLM(llm) for llm in llms]

    trainer = hydra.utils.instantiate(cfg.training, llm=llms[0], output_dir=cfg.paths.output_dir, logger=logger)

    data_queue = multiprocessing.Queue()
    task_queue = multiprocessing.JoinableQueue() if llm_support_async_calls else multiprocessing.Queue()
    workers = [
        EnvWorker(cfg, llms, task_queue, data_queue, logger, seed_shift + local_rank, seed + seed_shift + seed_gpu_wise)
        for seed_shift in range(cfg.training.config.env_workers)
    ]

    if cfg.training.config.env_workers > 1:
        for w in workers:
            w.start()

    while not trainer.done():
        start = time.time()
        for llm in llms:
            llm.start_generation()

        if cfg.training.config.env_workers > 1:
            # ask workers to generate
            for _ in workers:
                task_queue.put_nowait(None)

            if llm_support_async_calls:
                # wait for workers to finish generation
                task_queue.join()
            else:
                while data_queue.qsize() < cfg.training.config.env_workers:
                    for llm in llms:
                        llm.process_queue()

            total_steps, queries, responses, masks, all_rewards = zip(*[data_queue.get() for _ in workers])
        else:
            total_steps, queries, responses, masks, all_rewards = workers[0].collect_trajectories()
            total_steps, queries, responses, masks, all_rewards = (
                [total_steps],
                [queries],
                [responses],
                [masks],
                [all_rewards],
            )

        for llm in llms:
            llm.stop_generation()
        elapsed = time.time() - start
        total_steps = sum(total_steps)
        logger.log_metrics(
            {
                "total_env_steps": total_steps,
                "time_elapsed": elapsed,
                "avg_time_per_step": elapsed / total_steps,
                "avg_reward": np.mean(all_rewards),
            }
        )
        logger.save_metrics("all")
        trainer.train(queries, responses, masks, all_rewards, elapsed)

    for _ in workers:
        task_queue.put_nowait("stop")


if __name__ == "__main__":
    main()

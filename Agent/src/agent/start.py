import time
import traceback

import hydra
import numpy as np
import pyrootutils
from omegaconf import DictConfig
from omegaconf import OmegaConf

from agent import utils
from agent.loggers import ManyLoggers
from agent.loggers import Tag
from agent.memory import MemKey
from agent.utils import pylogger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = pylogger.get_pylogger(__name__)


def play_single_episode(
    cfg,
    agent,
    task,
    logger,
    subtask: str | None = None,
):
    obs = task.reset(subtask)
    agent.reset(task)
    agent.run_on_episode_start_flow()
    logger.log_metrics({"task_id": task.id})

    timestep = 0
    returns = 0
    done = False
    start_time = time.time()

    try:
        while not done and (cfg.max_env_steps is None or timestep < cfg.max_env_steps):
            logger.log_metrics({}, timestep=timestep)  # sets the timestep in the logger
            agent.observe(obs)

            agent.run_pre_action_flow()

            obs, rew, done = task.step(agent.external_action)
            agent.memory.store(rew, {MemKey.REWARD})

            agent.run_post_action_flow()
            logger.log_metrics({Tag.REWARD: rew, Tag.DONE: done})
            logger.save_metrics()
            timestep += 1
            returns += rew

        agent.run_on_episode_end_flow()
    except Exception:
        logger.log_metrics({Tag.ERROR: str(traceback.format_exc())})

    logger.log_metrics(
        {
            "discounted_returns": returns,
            "episode_timesteps": timestep,
            "episode_elapsed_time": time.time() - start_time,
        }
    )
    logger.save_metrics()

    return returns, timestep


@hydra.main(version_base="1.3", config_path="../../configs", config_name="default_sa_eval")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)
    OmegaConf.resolve(cfg)

    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )

    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    task = hydra.utils.instantiate(cfg.task)

    assert (
        isinstance(cfg.max_episodes, int) and cfg.max_episodes > 0
    ) or cfg.max_episodes is None, "max_episodes should be a positive integer or None."
    assert (
        isinstance(cfg.max_env_steps, int) and cfg.max_env_steps > 0
    ) or cfg.max_env_steps is None, "max_env_steps should be a positive integer or None."

    global_start_time = time.time()
    total_steps = 0
    all_returns = []

    agent.run_on_init_flow()

    while cfg.max_episodes is None or len(all_returns) < cfg.max_episodes:
        logger.log_metrics({}, episode=len(all_returns), timestep=0)  # sets the episode number in the logger

        returns, timesteps = play_single_episode(cfg, agent, task, logger)
        total_steps += timesteps
        all_returns.append(returns)

    elapsed = time.time() - global_start_time
    logger.log_metrics(
        {
            "total_env_steps": total_steps,
            "time_elapsed": elapsed,
            "avg_time_per_step": elapsed / total_steps if total_steps else None,
            "avg_returns": np.mean(all_returns),
        }
    )
    logger.save_metrics()


if __name__ == "__main__":
    main()

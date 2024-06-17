import json
import traceback
import uuid
from functools import partial

import hydra
import pyrootutils
import redis
from omegaconf import DictConfig
from omegaconf import OmegaConf

from agent import utils
from agent.loggers import APIUsageLogger
from agent.loggers import ManyLoggers
from agent.loggers import Tag
from agent.utils import pylogger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = pylogger.get_pylogger(__name__)


def play_single_episode(
    cfg,
    agent,
    task,
    logger,
    subtask,
):
    obs = task.reset(subtask)
    agent.reset(task)

    timestep = 0
    episode_reward = 0

    api_usage_logger = next((log for log in logger.loggers if isinstance(log, APIUsageLogger)), None)
    if api_usage_logger:
        api_usage_logger.reset_usage_metrics()

    while not task.is_complete() and (cfg.max_env_steps is None or timestep < cfg.max_env_steps):
        logger.log_metrics({}, timestep=timestep)  # sets the timestep in the logger
        agent_ready = False
        agent.observe(obs)
        thinking_count = 0
        while not agent_ready and (cfg.max_thinking_steps is None or thinking_count < cfg.max_thinking_steps):
            agent_ready = agent.step() if not agent.ready() else None
            thinking_count += 1
        else:
            agent.step() if not agent.ready() else None

        actions = {agent.name: agent.external_action}
        obs, rew, done = task.step(actions)
        logger.log_metrics({Tag.REWARD: rew, Tag.DONE: done})
        logger.save_metrics()
        timestep += 1
        episode_reward += rew

    if api_usage_logger:
        usage_dict = {
            "input_tokens": api_usage_logger.input_usage,
            "output_tokens": api_usage_logger.output_usage,
            "cost": api_usage_logger.get_cost(),
        }
    else:
        usage_dict = {"input_tokens": 0, "output_tokens": 0, "cost": 0}

    return {"discounted_returns": episode_reward, **usage_dict}


def listen_for_tasks(redis_client, agent_uuid, task_queue, result_queue, logger, callback):
    print("Agent waiting for tasks on queue: ", task_queue)
    episode_count = 0
    while True:
        task_data = redis_client.blpop(task_queue, timeout=30)

        if task_data:
            subtask = json.loads(task_data[1])["subtask"]
            print(f"Agent {agent_uuid} is starting task: {subtask}!")
            logger.log_metrics({}, timestep=0, episode=episode_count)  # sets the timestep in the logger
            try:
                result = callback(subtask)
            except Exception:
                logger.log_metrics({Tag.ERROR: str(traceback.format_exc())})
                print(traceback.format_exc())
                result = {
                    "discounted_returns": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0,
                }  # TODO: fix for negative reward.

            redis_client.rpush(result_queue, json.dumps({"agent": agent_uuid, "task": subtask, "result": result}))
            episode_count += 1
        else:
            print(f"Agent {agent_uuid} is waiting for tasks...")


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

    agent_uuid = uuid.uuid4().hex

    redis_client = redis.Redis(host="localhost", port=6379, db=0)
    # redis_client.delete("agents_pool")

    agent_attributes = {
        "task": cfg.task.name,
        "agent": {k: v for k, v in (item.split("=", 1) for item in cfg.experiment_name.split(","))},
    }
    callback = partial(play_single_episode, cfg, agent, task, logger)

    try:
        redis_client.hset("agents_pool", agent_uuid, json.dumps(agent_attributes, sort_keys=True))
        listen_for_tasks(
            redis_client,
            agent_uuid,
            f"task_queue:{agent_uuid}",
            "result_queue",
            logger,
            callback,
        )
    finally:
        redis_client.hdel("agents_pool", agent_uuid)


if __name__ == "__main__":
    main()

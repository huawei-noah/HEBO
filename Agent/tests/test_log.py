import hydra
import pyrootutils
from hydra import compose
from hydra import initialize
from omegaconf import OmegaConf
from omegaconf import open_dict

from agent.loggers import ManyLoggers


def test_log():
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="default_sa_eval.yaml",
            return_hydra_config=True,
            overrides=["method=fs-cot-sc", "task=gsm8k", "llm@agent.llm=random"],
        )
        with open_dict(cfg):
            cfg.paths.root_dir = str(pyrootutils.find_root(indicator=".project-root"))
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
    OmegaConf.resolve(cfg)

    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)

    agent.logger.log_metrics(
        {
            "bracket:test1": """Action: search[slim fit, machine wash women's jumpsuits, rompers &
                                overalls with short sleeve, high waist, polyester spandex, daily wear,
                                color: green stripe, size: large, price under 50.00 dollars]"""
        }
    )
    agent.logger.log_metrics({"bracket:test2": "asd [/floorlamp 1/poof]"})

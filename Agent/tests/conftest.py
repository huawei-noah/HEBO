"""This file prepares config fixtures for other tests."""

import pyrootutils
import pytest
from hydra import compose
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from omegaconf import open_dict


@pytest.fixture(scope="package")
def cfg_alfworld_global() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="default_sa_eval.yaml",
            return_hydra_config=True,
            overrides=["method=fs-cot", "task=alfworld", "llm@agent.llm=random", "logger=stdout_logger"],
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(pyrootutils.find_root(indicator=".project-root"))
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            # cfg.logger = None

    return cfg


# this is called by each test which uses `cfg_train` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_alfworld(cfg_alfworld_global, tmp_path) -> DictConfig:
    cfg = cfg_alfworld_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()

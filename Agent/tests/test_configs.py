from hydra import compose
from hydra import initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def test_config_alfworld_global(cfg_alfworld: DictConfig):
    assert cfg_alfworld
    assert cfg_alfworld.agent
    assert cfg_alfworld.task

    HydraConfig().set_config(cfg_alfworld)


# test environments


def test_config_alfworld_0() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="default_sa_eval.yaml",
            return_hydra_config=True,
            overrides=["method=fs-cot", "task=alfworld", "llm@agent.llm=random"],
        )

        assert cfg.task.name == "ALFWorld"
        assert cfg.task.train_eval == "eval_out_of_distribution"
        assert len(cfg.agent.prompt_builder.template_paths) == 2
        assert cfg.agent.prompt_builder.template_paths[0] == "alfworld"
        assert cfg.agent.prompt_builder.template_paths[1] == "default"


def test_config_gsm8k_0() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="default_sa_eval.yaml",
            return_hydra_config=True,
            overrides=["method=fs-cot", "task=gsm8k", "llm@agent.llm=random"],
        )

        assert cfg.task.name == "gsm8k"
        assert cfg.task.split == "test"
        assert len(cfg.agent.prompt_builder.template_paths) == 2
        assert cfg.agent.prompt_builder.template_paths[0] == "gsm8k"
        assert cfg.agent.prompt_builder.template_paths[1] == "default"


def test_config_hotpotqa_0() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="default_sa_eval.yaml",
            return_hydra_config=True,
            overrides=["method=fs-cot", "task=hotpotqa", "llm@agent.llm=random"],
        )

        assert cfg.task.name == "HotpotQA"
        assert cfg.task.split == "validation"
        assert len(cfg.agent.prompt_builder.template_paths) == 2
        assert cfg.agent.prompt_builder.template_paths[0] == "hotpotqa"
        assert cfg.agent.prompt_builder.template_paths[1] == "default"


def test_config_webshop_0() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="default_sa_eval.yaml",
            return_hydra_config=True,
            overrides=["method=fs-cot", "task=webshop", "llm@agent.llm=random"],
        )

        assert cfg.task.name == "WebShop"
        # assert cfg.task.train_eval == "eval_out_of_distribution"
        assert len(cfg.agent.prompt_builder.template_paths) == 2
        assert cfg.agent.prompt_builder.template_paths[0] == "webshop"
        assert cfg.agent.prompt_builder.template_paths[1] == "default"


def test_config_direct_0() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="default_sa_eval.yaml",
            return_hydra_config=True,
            overrides=["method=direct", "task=alfworld", "llm@agent.llm=random"],
        )

        assert cfg.agent.prompt_builder.default_kwargs["cot_type"] == "zero_shot"
        assert cfg.agent.pre_action_flow.sequence[0]._target_ == "agent.commands.Act"


def test_config_fs_0() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="default_sa_eval.yaml",
            return_hydra_config=True,
            overrides=["method=fs", "task=alfworld", "llm@agent.llm=random"],
        )

        assert cfg.agent.prompt_builder.default_kwargs["cot_type"] == "few_shot"
        assert cfg.agent.pre_action_flow.sequence[0]._target_ == "agent.commands.Act"


def test_config_zs_cot_0() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="default_sa_eval.yaml",
            return_hydra_config=True,
            overrides=["method=zs-cot", "task=alfworld", "llm@agent.llm=random"],
        )

        assert cfg.agent.prompt_builder.default_kwargs["cot_type"] == "zero_shot_cot"
        assert cfg.agent.pre_action_flow.sequence[0]._target_ == "agent.commands.Act"


def test_config_fs_cot_0() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="default_sa_eval.yaml",
            return_hydra_config=True,
            overrides=["method=fs-cot", "task=alfworld", "llm@agent.llm=random"],
        )

        assert cfg.agent.prompt_builder.default_kwargs["cot_type"] == "few_shot_cot"
        assert cfg.agent.pre_action_flow.sequence[0]._target_ == "agent.commands.Act"


def test_config_fs_cot_sc_0() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="default_sa_eval.yaml",
            return_hydra_config=True,
            overrides=["method=fs-cot-sc", "task=alfworld", "llm@agent.llm=random"],
        )

        assert cfg.agent.prompt_builder.default_kwargs["cot_type"] == "few_shot_cot"
        assert cfg.agent.pre_action_flow.sequence[0]._target_ == "agent.commands.SelfConsistencyAct"


def test_config_fs_cot_react_0() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="default_sa_eval.yaml",
            return_hydra_config=True,
            overrides=["method=fs-cot-react", "task=alfworld", "llm@agent.llm=random"],
        )

        assert cfg.agent.prompt_builder.default_kwargs["cot_type"] == "react"
        assert cfg.agent.pre_action_flow.choices[0].sequence[0]._target_ == "agent.commands.Think"
        assert cfg.agent.pre_action_flow.choices[0].sequence[1]._target_ == "agent.commands.Act"
        assert cfg.agent.pre_action_flow.choices[1]._target_ == "agent.commands.Act"

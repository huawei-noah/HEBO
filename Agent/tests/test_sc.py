import hydra
import pyrootutils
from hydra import compose
from hydra import initialize
from omegaconf import OmegaConf
from omegaconf import open_dict

from agent.commands import SelfConsistencyAct
from agent.loggers import ManyLoggers
from agent.memory import MemKey
from agent.tasks import ActionSpace
from agent.tasks import Task

default_task_param = {"name": "test", "subtask": None, "version": "1.0", "description": "test", "agent": ""}


class TestTask(Task):
    def __init__(self, action_space, parsing, answer_format, **kwargs):
        super().__init__(**kwargs)

        self.action_space = action_space
        self.parsing = parsing
        self.answer_format = answer_format

    def reset(self) -> None:
        self.done = False

    def get_observation(self, agent):
        pass

    @property
    def answer_parser(self):
        return self.parsing

    def is_complete(self):
        return self.done

    def step(self, actions):
        pass


def test_sc_select():
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="default_sa_eval.yaml",
            return_hydra_config=True,
            overrides=["method=fs-cot-sc", "task=alfworld", "llm@agent.llm=random"],
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
    # task = hydra.utils.instantiate(cfg.task)
    # agent.memory.store(["PLACEHOLDER"], {"available_actions"})
    # agent.available_actions = "PLACEHOLDER"
    # agent.task = task

    agent.memory.store(["PLACEHOLDER"], {MemKey.AVAILABLE_ACTIONS})
    agent.available_actions = "PLACEHOLDER"
    agent.task = TestTask(action_space=ActionSpace.DISCRETE, parsing="Action", answer_format="", **default_task_param)

    sc = SelfConsistencyAct()  # choose the one most frequent, if frequency the same, choose the first one

    def _thunk(agent):
        for subcommand in sc.sequence[:-1]:
            subcommand(agent)
            agent.memory.retrieve({MemKey.NEXT_PLANNED_ACTION_DIVERSE: 1.0})

    sc.select_action = _thunk

    responses = iter(["action c", "PLACEHOLDER", "action a", "action a", "action a"])
    agent.llm.choose_from_options = lambda messages, options, parse_func: next(responses)
    sc.select_action(agent)
    assert agent.memory.retrieve({MemKey.NEXT_PLANNED_ACTION: 1.0}) == "action a"
    agent.memory.reset()

    agent.memory.store(["PLACEHOLDER"], {MemKey.AVAILABLE_ACTIONS})
    responses = iter(["action c", "PLACEHOLDER", "action a", "action a", "action c"])
    agent.llm.choose_from_options = lambda messages, options, parse_func: next(responses)
    sc.select_action(agent)
    assert agent.memory.retrieve({MemKey.NEXT_PLANNED_ACTION: 1.0}) in ["action a", "action c"]
    agent.memory.reset()

    agent.memory.store(["PLACEHOLDER"], {MemKey.AVAILABLE_ACTIONS})
    responses = iter(["action a", "PLACEHOLDER", "action c", "action a", "action c"])
    agent.llm.choose_from_options = lambda messages, options, parse_func: next(responses)
    sc.select_action(agent)
    assert agent.memory.retrieve({MemKey.NEXT_PLANNED_ACTION: 1.0}) in ["action a", "action c"]
    agent.memory.reset()

    agent.memory.store(["PLACEHOLDER"], {MemKey.AVAILABLE_ACTIONS})
    responses = iter(["action a", "action a", "action a", "action a", "action a"])
    agent.llm.choose_from_options = lambda messages, options, parse_func: next(responses)
    sc.select_action(agent)
    assert agent.memory.retrieve({MemKey.NEXT_PLANNED_ACTION: 1.0}) == "action a"
    agent.memory.reset()

    agent.memory.store(["PLACEHOLDER"], {MemKey.AVAILABLE_ACTIONS})
    responses = iter(["action e", "action d", "action c", "action b", "action a"])
    agent.llm.choose_from_options = lambda messages, options, parse_func: next(responses)
    sc.select_action(agent)
    assert agent.memory.retrieve({MemKey.NEXT_PLANNED_ACTION: 1.0}) in [
        "action e",
        "action d",
        "action c",
        "action b",
        "action a",
    ]


def test_sc_generate_numerical():
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
    task = hydra.utils.instantiate(cfg.task)
    agent.memory.store(["PLACEHOLDER"], {MemKey.AVAILABLE_ACTIONS})
    agent.available_actions = "PLACEHOLDER"
    agent.task = task

    sc = SelfConsistencyAct()  # choose the one most frequent, if frequency the same, choose the first one

    def _thunk(agent):
        for subcommand in sc.sequence[:-1]:
            subcommand(agent)
            agent.memory.retrieve({MemKey.NEXT_PLANNED_ACTION_DIVERSE: 1.0})

    sc.generate_action = _thunk

    responses = iter(["916", "5", "", "329", "329"])
    agent.llm.chat_completion = lambda messages, parse_func: next(responses)
    sc.generate_action(agent)
    assert agent.memory.retrieve({MemKey.NEXT_PLANNED_ACTION: 1.0}) in ["329"]
    agent.memory.reset()

    responses = iter(["916", "916", "", "329", "329"])
    agent.llm.chat_completion = lambda messages, parse_func: next(responses)
    sc.generate_action(agent)
    assert agent.memory.retrieve({MemKey.NEXT_PLANNED_ACTION: 1.0}) in [
        "916",
        "329",
    ]
    agent.memory.reset()

    responses = iter(["916", "5", "", "329", "329"])
    agent.llm.chat_completion = lambda messages, parse_func: next(responses)
    sc.generate_action(agent)
    assert agent.memory.retrieve({MemKey.NEXT_PLANNED_ACTION: 1.0}) in ["329"]
    agent.memory.reset()

    responses = iter(["916", "", "", "", ""])
    agent.llm.chat_completion = lambda messages, parse_func: next(responses)
    sc.generate_action(agent)
    assert agent.memory.retrieve({MemKey.NEXT_PLANNED_ACTION: 1.0}) in ["916"]
    agent.memory.reset()


def test_sc_generate_text():
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="default_sa_eval.yaml",
            return_hydra_config=True,
            overrides=["method=fs-cot-sc", "task=hotpotqa", "llm@agent.llm=random"],
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
    task = hydra.utils.instantiate(cfg.task)
    agent.memory.store("PLACEHOLDER", {MemKey.AVAILABLE_ACTIONS})
    agent.available_actions = "PLACEHOLDER"
    agent.task = task

    sc = SelfConsistencyAct()  # choose the one most similar to others, if similarity the same, choose the first one

    def _thunk(agent):
        for subcommand in sc.sequence[:-1]:
            subcommand(agent)
            agent.memory.retrieve({MemKey.NEXT_PLANNED_ACTION_DIVERSE: 1.0})

    sc.generate_action = _thunk

    responses = iter(
        [
            "Animorphs",
            "The Divide trilogy",
            "The Hork-Bajir Chronicles",
            "The Animorphs series",
            "Animorphs",
        ]
    )
    agent.llm.chat_completion = lambda messages, parse_func: next(responses)
    sc.generate_action(agent)
    assert agent.memory.retrieve({MemKey.NEXT_PLANNED_ACTION: 1.0}) == "Animorphs"
    agent.memory.reset()

    responses = iter(
        [
            "None",
            "None",
            "No relevant government position",
            "Secretary of State for",
            "Voice actress",
        ]
    )
    agent.llm.chat_completion = lambda messages, parse_func: next(responses)
    sc.generate_action(agent)
    assert agent.memory.retrieve({MemKey.NEXT_PLANNED_ACTION: 1.0}) == "None"
    agent.memory.reset()

    responses = iter(["Yes", "Yes", "No", "Yes", "Yes"])
    agent.llm.chat_completion = lambda messages, parse_func: next(responses)
    sc.generate_action(agent)
    assert agent.memory.retrieve({MemKey.NEXT_PLANNED_ACTION: 1.0}) == "Yes"
    agent.memory.reset()

    responses = iter(["... Yes", "Yes", "No", "Yes", "Yes"])
    agent.llm.chat_completion = lambda messages, parse_func: next(responses)
    sc.generate_action(agent)
    assert agent.memory.retrieve({MemKey.NEXT_PLANNED_ACTION: 1.0}) == "Yes"
    agent.memory.reset()

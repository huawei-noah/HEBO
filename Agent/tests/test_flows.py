import hydra
from omegaconf import OmegaConf

from agent.commands import ConsiderAction
from agent.commands import DecisionFlow
from agent.commands import LoopFlow
from agent.commands import SequentialFlow
from agent.commands import Think
from agent.loggers import ManyLoggers
from tests.utils.random_llm import RandomLanguageBackend


def test_flow_sequential_simple_0(cfg_alfworld):
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    # not needed:
    # task = hydra.utils.instantiate(cfg.task, agents=[agents[i] for i in cfg.task.agents])

    cmds = (Think(),)
    s1 = SequentialFlow(cmds)

    assert s1.step(agent) is cmds[0]

    assert s1.step(agent) is None


def test_flow_sequential_simple_1(cfg_alfworld):
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    # not needed:
    # task = hydra.utils.instantiate(cfg.task, agents=[agents[i] for i in cfg.task.agents])

    cmds = t1, t2, t3, t4 = [Think(name="think1"), Think(name="think2"), Think(name="think3"), ConsiderAction()]
    s1 = SequentialFlow(cmds)

    for t in cmds:
        assert s1.step(agent) is t

    assert s1.step(agent) is None


def test_flow_sequential_nested_0(cfg_alfworld):
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    # not needed:
    # task = hydra.utils.instantiate(cfg.task, agents=[agents[i] for i in cfg.task.agents])

    t1, t2, t3, t4 = [Think(name="think1"), Think(name="think2"), Think(name="think3"), ConsiderAction()]

    s1 = SequentialFlow([t1, t2, t3, t4])

    s2 = SequentialFlow([s1, t4, t3, t2, t1])

    assert s2.step(agent) is t1
    assert s2.step(agent) is t2
    assert s2.step(agent) is t3
    assert s2.step(agent) is t4

    assert s2.step(agent) is t4
    assert s2.step(agent) is t3
    assert s2.step(agent) is t2
    assert s2.step(agent) is t1

    assert s2.step(agent) is None


def test_flow_sequential_nested_1(cfg_alfworld):
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    # not needed:
    # task = hydra.utils.instantiate(cfg.task, agents=[agents[i] for i in cfg.task.agents])

    t1, t2, t3, t4 = [Think(name="think1"), Think(name="think2"), Think(name="think3"), ConsiderAction()]

    s1 = SequentialFlow([t1, t2, t3, t4])
    s2 = SequentialFlow([t4, t3, t2, t1])

    s3 = SequentialFlow([s1, s2])

    assert s3.step(agent) is t1
    assert s3.step(agent) is t2
    assert s3.step(agent) is t3
    assert s3.step(agent) is t4

    assert s3.step(agent) is t4
    assert s3.step(agent) is t3
    assert s3.step(agent) is t2
    assert s3.step(agent) is t1

    assert s3.step(agent) is None


def test_flow_sequential_nested_2(cfg_alfworld):
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    # not needed:
    # task = hydra.utils.instantiate(cfg.task, agents=[agents[i] for i in cfg.task.agents])

    t1, t2, t3, t4 = [Think(name="think1"), Think(name="think2"), Think(name="think3"), ConsiderAction()]

    s1 = SequentialFlow([t1, t2, t3, t4])
    s2 = SequentialFlow([t4, t3, t2, t1])

    s3 = SequentialFlow([s1, t1, t1, s2, t1])

    assert s3.step(agent) is t1
    assert s3.step(agent) is t2
    assert s3.step(agent) is t3
    assert s3.step(agent) is t4

    assert s3.step(agent) is t1
    assert s3.step(agent) is t1

    assert s3.step(agent) is t4
    assert s3.step(agent) is t3
    assert s3.step(agent) is t2
    assert s3.step(agent) is t1

    assert s3.step(agent) is t1

    assert s3.step(agent) is None


def test_flow_sequential_nested_3(cfg_alfworld):
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    # not needed:
    # task = hydra.utils.instantiate(cfg.task, agents=[agents[i] for i in cfg.task.agents])

    cmds = t1, t2, t3, t4 = [Think(name="think1"), Think(name="think2"), Think(name="think3"), ConsiderAction()]

    s1 = SequentialFlow([t1, t2, t3, t4])
    s2 = SequentialFlow([t1, t2, t3, t4])

    s3 = SequentialFlow([s1, s2])

    s4 = SequentialFlow([t1, s3, t1])

    assert s4.step(agent) is t1

    for cmd in cmds:
        assert s4.step(agent) is cmd

    for cmd in cmds:
        assert s4.step(agent) is cmd

    assert s4.step(agent) is t1

    assert s4.step(agent) is None


def test_flow_decision_simple_0(cfg_alfworld):
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    agent.llm = RandomLanguageBackend()
    agent.llm.choose_from_options = lambda messages, options, parse_func: options[0]
    # not needed:
    # task = hydra.utils.instantiate(cfg.task, agents=[agents[i] for i in cfg.task.agents])

    cmds = (Think(), ConsiderAction())
    d1 = DecisionFlow(cmds)

    assert d1.step(agent) is cmds[0]
    assert d1.step(agent) is None


def test_flow_decision_simple_1(cfg_alfworld):
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    agent.llm = RandomLanguageBackend()
    agent.llm.choose_from_options = lambda messages, options, parse_func: options[1]
    # not needed:
    # task = hydra.utils.instantiate(cfg.task, agents=[agents[i] for i in cfg.task.agents])

    cmds = (Think(), ConsiderAction())
    d1 = DecisionFlow(cmds)

    assert d1.step(agent) is cmds[1]
    assert d1.step(agent) is None


def test_flow_decision_simple_2(cfg_alfworld):
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    agent.llm = RandomLanguageBackend()
    agent.llm.choose_from_options = lambda messages, options, parse_func: options[2]
    # not needed:
    # task = hydra.utils.instantiate(cfg.task, agents=[agents[i] for i in cfg.task.agents])

    cmds = (Think(), ConsiderAction(), Think(name="think2"))
    d1 = DecisionFlow(cmds)

    assert d1.step(agent) is cmds[2]
    assert d1.step(agent) is None


def test_flow_decision_nested_0(cfg_alfworld):
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    agent.llm = RandomLanguageBackend()
    agent.llm.choose_from_options = lambda messages, options, parse_func: options[0]
    # not needed:
    # task = hydra.utils.instantiate(cfg.task, agents=[agents[i] for i in cfg.task.agents])

    cmds = t1, t2, t3 = (Think(), ConsiderAction(), Think(name="think2"))
    d1 = DecisionFlow(cmds)

    d2 = DecisionFlow([d1, t3])

    assert d2.step(agent) is t1
    assert d2.step(agent) is None


def test_flow_decision_nested_1(cfg_alfworld):
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    agent.llm = RandomLanguageBackend()
    agent.llm.choose_from_options = lambda messages, options, parse_func: options[1]
    # not needed:
    # task = hydra.utils.instantiate(cfg.task, agents=[agents[i] for i in cfg.task.agents])

    cmds = t1, t2, t3 = (Think(), ConsiderAction(), Think(name="think2"))
    d1 = DecisionFlow(cmds)

    d2 = DecisionFlow([d1, t3])

    assert d2.step(agent) is t3
    assert d2.step(agent) is None


def test_flow_decision_nested_2(cfg_alfworld):
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    agent.llm = RandomLanguageBackend()
    agent.llm.choose_from_options = lambda messages, options, parse_func: options[0]
    # not needed:
    # task = hydra.utils.instantiate(cfg.task, agents=[agents[i] for i in cfg.task.agents])

    cmds = t1, t2, t3 = (Think(), ConsiderAction(), Think(name="think2"))
    d1 = DecisionFlow(cmds)
    d2 = DecisionFlow([d1, t3])
    DecisionFlow([d2, t3])

    assert d1.step(agent) is t1
    assert d1.step(agent) is None


def test_flow_decision_combined_0(cfg_alfworld):
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    agent.llm = RandomLanguageBackend()
    agent.llm.choose_from_options = lambda messages, options, parse_func: options[0]
    # not needed:
    # task = hydra.utils.instantiate(cfg.task, agents=[agents[i] for i in cfg.task.agents])

    cmds = t1, t2, t3 = (Think(), ConsiderAction(), Think(name="think2"))
    s1 = SequentialFlow(cmds)
    s2 = SequentialFlow(cmds[::-1], name="sequence2")

    d1 = DecisionFlow([s1, s2])

    assert d1.step(agent) is t1
    assert d1.step(agent) is t2
    assert d1.step(agent) is t3
    assert d1.step(agent) is None
    d1.reset()
    assert d1.step(agent) is t1
    assert d1.step(agent) is t2
    assert d1.step(agent) is t3
    assert d1.step(agent) is None


def test_flow_decision_combined_1(cfg_alfworld):
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    agent.llm = RandomLanguageBackend()
    agent.llm.choose_from_options = lambda messages, options, parse_func: options[0]
    # not needed:
    # task = hydra.utils.instantiate(cfg.task, agents=[agents[i] for i in cfg.task.agents])

    t1, t2, t3 = (Think(), ConsiderAction(), Think(name="think2"))
    d1 = DecisionFlow([t1, t2])

    s1 = SequentialFlow([d1, t3])

    assert s1.step(agent) is t1
    assert s1.step(agent) is t3
    assert s1.step(agent) is None

    s1.reset()
    assert s1.step(agent) is t1
    assert s1.step(agent) is t3
    assert s1.step(agent) is None


def test_flow_loop_simple_0(cfg_alfworld):
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    # not needed:
    # task = hydra.utils.instantiate(cfg.task, agents=[agents[i] for i in cfg.task.agents])

    cmds = (Think(),)
    l1 = LoopFlow(loop_body=cmds[0], max_repetitions=3, allow_early_break=False)
    l1.reset()
    assert l1.step(agent) is cmds[0]
    assert l1.step(agent) is cmds[0]
    assert l1.step(agent) is cmds[0]
    assert l1.step(agent) is None


def test_flow_loop_simple_1(cfg_alfworld):
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    agent.llm = RandomLanguageBackend()
    agent.llm.choose_from_options = lambda messages, options, parse_func: options[1]
    # not needed:
    # task = hydra.utils.instantiate(cfg.task, agents=[agents[i] for i in cfg.task.agents])

    cmds = (Think(),)
    l1 = LoopFlow(loop_body=cmds[0], max_repetitions=4, allow_early_break=True)
    l1.reset()
    assert l1.step(agent) is cmds[0]
    assert l1.step(agent) is cmds[0]
    agent.llm.choose_from_options = lambda messages, options, parse_func: options[0]
    assert l1.step(agent) is None


def test_flow_loop_sequence_0(cfg_alfworld):
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)

    t1, t2 = (Think(), ConsiderAction())
    l1 = LoopFlow(loop_body=t1, max_repetitions=2, allow_early_break=False)

    s1 = SequentialFlow([l1, t2])
    s1.reset()
    assert s1.step(agent) is t1
    assert s1.step(agent) is t1
    assert s1.step(agent) is t2
    assert s1.step(agent) is None


def test_flow_loop_sequence_1(cfg_alfworld):
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)

    t1, t2 = (Think(), ConsiderAction())
    l1 = LoopFlow(loop_body=t1, max_repetitions=2, allow_early_break=False)
    l2 = LoopFlow(loop_body=t1, max_repetitions=2, allow_early_break=False)

    s1 = SequentialFlow([l1, t2, l2])
    s1.reset()
    assert s1.step(agent) is t1
    assert s1.step(agent) is t1
    assert s1.step(agent) is t2
    assert s1.step(agent) is t1
    assert s1.step(agent) is t1
    assert s1.step(agent) is None

    s1.reset()
    assert s1.step(agent) is t1
    assert s1.step(agent) is t1
    assert s1.step(agent) is t2
    assert s1.step(agent) is t1
    assert s1.step(agent) is t1
    assert s1.step(agent) is None


def test_flow_loop_decision_0(cfg_alfworld):
    cfg = cfg_alfworld
    OmegaConf.resolve(cfg)
    logger = ManyLoggers(
        loggers=[hydra.utils.instantiate(logger, project_cfg=cfg, _recursive_=False) for logger in cfg.logger.values()]
    )
    agent = hydra.utils.instantiate(cfg.agent, logger=logger)
    agent.llm = RandomLanguageBackend()
    agent.llm.choose_from_options = lambda messages, options, parse_func: options[0]

    t1, t2 = (Think(), ConsiderAction())
    l1 = LoopFlow(loop_body=t1, max_repetitions=2, allow_early_break=False, name="loop1")
    s1 = SequentialFlow([l1, t2], name="sequence1")
    d1 = DecisionFlow([s1, l1])
    l2 = LoopFlow(loop_body=d1, max_repetitions=2, allow_early_break=False)
    l2.reset()

    assert l2.step(agent) is t1
    assert l2.step(agent) is t1
    assert l2.step(agent) is t2
    agent.llm.choose_from_options = lambda messages, options, parse_func: options[1]
    assert l2.step(agent) is t1
    assert l2.step(agent) is t1
    assert l2.step(agent) is None

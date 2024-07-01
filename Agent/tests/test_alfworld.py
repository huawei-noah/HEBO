import alfworld.agents.environment as environment
import numpy as np
import pytest
import yaml


@pytest.mark.skip(reason="Needs to be fixed.")
def test_alfworld():
    with open("tests/base_config.yaml") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    env_type = config["env"]["type"]  # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

    # setup environment
    env = getattr(environment, env_type)(config, train_eval="train")
    env = env.init_env(batch_size=1)

    # interact
    obs, info = env.reset()
    # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
    admissible_commands = list(
        info["admissible_commands"]
    )  # note: BUTLER generates commands word-by-word without using admissible_commands
    random_actions = [np.random.choice(admissible_commands[0])]
    # step
    obs, scores, dones, infos = env.step(random_actions)
    print(f"Action: {random_actions[0]}, Obs: {obs[0]}")

from safe_rl.pg.agents import CPOAgent
from tf_algos.safety_starter_agents.run_agents import run_polopt_agent


def cpo(**kwargs):
    """Set up to run CPO."""
    cpo_kwargs = dict(
                    reward_penalized=False,  # Irrelevant in CPO
                    objective_penalized=False,  # Irrelevant in CPO
                    learn_penalty=False,  # Irrelevant in CPO
                    penalty_param_loss=False  # Irrelevant in CPO
                    )
    agent = CPOAgent(**cpo_kwargs)
    run_polopt_agent(agent=agent, **kwargs)
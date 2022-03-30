from safe_rl.pg.agents import TRPOAgent
from tf_algos.safety_starter_agents.run_agents import run_polopt_agent


def trpo(**kwargs):
    """Run Vanilla TRPO Lagrangian."""  
    trpo_kwargs = dict(
                    reward_penalized=False,
                    objective_penalized=False,
                    learn_penalty=False,
                    penalty_param_loss=False  # Irrelevant in unconstrained
                    )
    agent = TRPOAgent(**trpo_kwargs)
    run_polopt_agent(agent=agent, **kwargs)


def saute_trpo(**kwargs):
    """Run Saute TRPO."""  
    trpo_kwargs = dict(
                    reward_penalized=False,
                    objective_penalized=False,
                    learn_penalty=False,
                    penalty_param_loss=False  # Irrelevant in unconstrained
                    )
    agent = TRPOAgent(**trpo_kwargs)        
    kwargs['saute_constraints'] = True
    run_polopt_agent(agent=agent, **kwargs)


def trpo_lagrangian(**kwargs):
    """Run TRPO Lagrangian."""  
    # Objective-penalized form of Lagrangian TRPO.
    trpo_kwargs = dict(
                    reward_penalized=False,
                    objective_penalized=True,
                    learn_penalty=True,
                    penalty_param_loss=True,
                    backtrack_iters=kwargs['backtrack_iters']
                    )
    agent = TRPOAgent(**trpo_kwargs)
    run_polopt_agent(agent=agent, **kwargs)


def trpo_cvar(**kwargs):
    """
    Set up to run TRPO Lagrangian
    """
    # Objective-penalized form of Lagrangian TRPO.
    trpo_kwargs = dict(
                    reward_penalized=False,
                    objective_penalized=True,
                    learn_penalty=True,
                    penalty_param_loss=True,
                    backtrack_iters=kwargs['backtrack_iters']
                    )
    agent = TRPOAgent(**trpo_kwargs)
    kwargs['CVaR'] = True
    run_polopt_agent(agent=agent, **kwargs)


def saute_trpo_lagrangian(**kwargs):
    """Run Saute TRPO Lagrangian"""  
    # Objective-penalized form of Lagrangian PPO.
    trpo_kwargs = dict(
                    reward_penalized=False,
                    objective_penalized=True,
                    learn_penalty=True,
                    penalty_param_loss=True
                    )
    agent = TRPOAgent(**trpo_kwargs)
    kwargs['saute_lagrangian'] = True
    run_polopt_agent(agent=agent, **kwargs)
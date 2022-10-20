"""
SAC Agent and the run file. Copied from safety starter agents with minor modifications
"""
from typing import Callable
import numpy as np
import tensorflow as tf
import time
from safe_rl.utils.logx import EpochLogger
from safe_rl.utils.mpi_tf import sync_all_params, MpiAdamOptimizer
from safe_rl.utils.mpi_tools import mpi_fork, mpi_sum, proc_id, num_procs
from .sac_utils import mlp_actor, mlp_var, mlp_critic, placeholders, ReplayBuffer, count_vars, get_vars, get_target_update
from scipy.stats import norm
from collections import deque
import pandas as pd
import os 


sac_cfg = dict(
        log=True, 
        log_updates=True,         
        seed=42,
        steps_per_epoch=1000, 
        epochs=1000, 
        replay_size=int(1e6), 
        discount_factor=0.99,
        safety_discount_factor=0.99,
        safety_budget=1.0,         
        tau=0.005,
        lr=5e-4,
        alpha_lr=5e-4,
        penalty_lr=5e-2,
        batch_size=1024, 
        local_start_steps=int(1e3),
        # max_ep_len=0, # removed as it should be taken from env_cfg
        checkpoint_frequency=10, 
        n_test_episodes=100,
        n_train_episodes=100,
        local_update_after=int(1e3),
        train_frequency=1, 
        test_frequency=0,
        render=False, 
        fixed_entropy_bonus=None, 
        target_entropy=-1.0,
        initial_alpha=0.0,
        fixed_cost_penalty=None, 
        cost_constraint=None, 
        use_mean_constraints=False,
        saute_lagrangian=False,
        reward_scale=1,
        use_cvar_constraints=False,
        damp_scale=0.0,
)

def vanilla_sac(
    **kwargs
):
    """Run Vanilla SAC."""  

    kwargs['fixed_cost_penalty'] = None 
    kwargs['cost_constraint'] = None 
    kwargs['safety_budget'] = None 
    kwargs['saute_constraints'] = False 
    kwargs['use_cvar_constraints'] = False   
    run_sac_algo(**kwargs)


def lagrangian_sac(
    **kwargs
):  
    """Run SAC Lagrangian."""  
    assert kwargs['safety_budget'] is not None
    kwargs['fixed_cost_penalty'] = None 
    kwargs['cost_constraint'] = None 
    kwargs['saute_constraints'] = False 
    kwargs['use_cvar_constraints'] = False   
    kwargs['use_mean_constraints'] = True   
    run_sac_algo(**kwargs)


def saute_sac(
    **kwargs
):
    """Run Saute SAC."""  
    assert kwargs['safety_budget'] is not None
    kwargs['fixed_cost_penalty'] = None 
    kwargs['cost_constraint'] = None 
    kwargs['saute_constraints'] = True   
    kwargs['use_cvar_constraints'] = False    
    kwargs['use_mean_constraints'] = False    
    run_sac_algo(**kwargs)

def wc_sac(
    **kwargs
):
    """Run Worst Case SAC from https://github.com/AlgTUDelft/WCSAC, which is based on safety starter agents https://github.com/openai/safety-starter-agents. """  
    assert kwargs['safety_budget'] is not None and kwargs['safety_budget'] > 0
    kwargs['fixed_cost_penalty'] = None 
    kwargs['cost_constraint'] = None 
    kwargs['saute_constraints'] = False 
    kwargs['use_cvar_constraints'] = True         
    kwargs['use_mean_constraints'] = False    
    raise NotImplementedError("Due to licencing issues we cannot release this part, but modifications from https://github.com/AlgTUDelft/WCSAC can be easily adapted.")

def saute_lagrangian_sac(
    **kwargs
):
    """Set up to run Saute SAC Lagrangian."""  

    assert kwargs['safety_budget'] is not None
    kwargs['fixed_cost_penalty'] = None 
    kwargs['cost_constraint'] = None 
    kwargs['saute_constraints'] = True   
    kwargs['saute_lagrangian'] = True       
    kwargs['use_cvar_constraints'] = False    
    kwargs['use_mean_constraints'] = False    
    run_sac_algo(**kwargs)

def run_sac_algo(
        env_name:str="",
        log:bool=True, 
        log_updates:bool=True,         
        train_env_fn:Callable=None, 
        test_env_fn:Callable=None, 
        actor_fn=mlp_actor, 
        critic_fn=mlp_critic, 
        var_fn=mlp_var,
        ac_kwargs=dict(), 
        seed=42,
        steps_per_epoch=1000, 
        epochs=100, 
        replay_size=int(1e6), 
        discount_factor=0.99,
        safety_discount_factor=0.99,
        tau=0.005,
        lr=1e-4, 
        alpha_lr=1e-4,
        penalty_lr=1e-5,
        batch_size=1024, 
        local_start_steps=int(1e3),
        max_ep_len=0, 
        logger_kwargs=dict(), 
        checkpoint_frequency=10, 
        n_test_episodes=10,
        n_train_episodes=10,
        local_update_after=int(1e3),
        train_frequency=1, 
        test_frequency=0, 
        render=False, 
        fixed_entropy_bonus=None, 
        target_entropy=-1.0,
        initial_alpha=0.0,
        fixed_cost_penalty=None, 
        cost_constraint=None,
        saute_constraints:bool=False,
        saute_lagrangian:bool=False,
        use_mean_constraints:bool=False,
        safety_budget=None,
        reward_scale=1,
        writer=None,
        use_cvar_constraints=False, # removed from repo
        damp_scale=0.0,             # removed from repo  
        cl=0.5                      # removed from repo  
    ):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_fn: A function which takes in placeholder symbols
            for state, ``x_ph``, and action, ``a_ph``, and returns the actor
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ===========  ================  ======================================

        critic_fn: A function which takes in placeholder symbols
            for state, ``x_ph``, action, ``a_ph``, and policy ``pi``,
            and returns the critic outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``critic``    (batch,)         | Gives one estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``critic_pi`` (batch,)         | Gives another estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_fn / critic_fn
            function you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        tau (float): Interpolation factor in tau averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak=1-tau. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        batch_size (int): Minibatch size for SGD.

        local_start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        checkpoint_frequency (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        fixed_entropy_bonus (float or None): Fixed bonus to reward for entropy.
            Units are (points of discounted sum of future reward) / (nats of policy entropy).
            If None, use ``entropy_constraint`` to set bonus value instead.

        entropy_constraint (float): If ``fixed_entropy_bonus`` is None,
            Adjust entropy bonus to maintain at least this much entropy.
            Actual constraint value is multiplied by the dimensions of the action space.
            Units are (nats of policy entropy) / (action dimenson).

        fixed_cost_penalty (float or None): Fixed penalty to reward for cost.
            Units are (points of discounted sum of future reward) / (points of discounted sum of future costs).
            If None, use ``cost_constraint`` to set penalty value instead.

        cost_constraint (float or None): If ``fixed_cost_penalty`` is None,
            Adjust cost penalty to maintain at most this much cost.
            Units are (points of discounted sum of future costs).
            Note: to get an approximate cost_constraint from a cost_lim (undiscounted sum of costs),
            multiply cost_lim by (1 - gamma ** episode_len) / (1 - gamma).
            If None, use cost_lim to calculate constraint.

        cost_lim (float or None): If ``cost_constraint`` is None,
            calculate an approximate constraint cost from this cost limit.
            Units are (expectation of undiscounted sum of costs in a single episode).
            If None, cost_lim is not used, and if no cost constraints are used, do naive optimization.
    """
    assert 0 <= discount_factor <= 1
    assert 0 <= safety_discount_factor <= 1
    use_costs = fixed_cost_penalty or cost_constraint or use_mean_constraints or saute_lagrangian
    polyak = 1 - tau
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Env instantiation
    env, test_env = train_env_fn(), test_env_fn()
    observation_space = env.observation_space 
    obs_dim = observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Setting seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    test_env.seed(seed)

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph, c_ph = placeholders(obs_dim, act_dim, obs_dim, None, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, pi, logp_pi = actor_fn(x_ph, a_ph, **ac_kwargs)
        qr1, qr1_pi = critic_fn(x_ph, a_ph, pi, name='qr1', **ac_kwargs)
        qr2, qr2_pi = critic_fn(x_ph, a_ph, pi, name='qr2', **ac_kwargs)
        qc, qc_pi = critic_fn(x_ph, a_ph, pi, name='qc', **ac_kwargs)

    with tf.variable_scope('main', reuse=True):
        # Additional policy output from a different observation placeholder
        # This lets us do separate optimization updates (actor, critics, etc)
        # in a single tensorflow op.
        _, pi2, logp_pi2 = actor_fn(x2_ph, a_ph, **ac_kwargs)

    # Target value network
    with tf.variable_scope('target'):
        _, qr1_pi_targ = critic_fn(x2_ph, a_ph, pi2, name='qr1', **ac_kwargs)
        _, qr2_pi_targ = critic_fn(x2_ph, a_ph, pi2, name='qr2', **ac_kwargs)
        _, qc_pi_targ = critic_fn(x2_ph, a_ph, pi2, name='qc', **ac_kwargs)


    # Entropy bonus
    if fixed_entropy_bonus is None:
        with tf.variable_scope('entreg'):
            soft_alpha = tf.get_variable('soft_alpha',
                                         initializer=initial_alpha,
                                         trainable=True,
                                         dtype=tf.float32)
        alpha = tf.nn.softplus(soft_alpha)
    else:
        alpha = tf.constant(fixed_entropy_bonus)
    # log_alpha = tf.log(alpha)
    log_alpha = tf.log(tf.clip_by_value(alpha, 1e-8, 1e8)) # clipping 

    # Cost penalty
    if use_costs:
        if fixed_cost_penalty is None:
            with tf.variable_scope('costpen'):
                soft_beta = tf.get_variable('soft_beta',
                                             initializer=0.0,
                                             trainable=True,
                                             dtype=tf.float32)
            beta = tf.nn.softplus(soft_beta)
        else:
            beta = tf.constant(fixed_cost_penalty)
        # log_beta = tf.log(beta)
        log_beta = tf.log(tf.clip_by_value(beta, 1e-4, 1e8)) # clipping beta
    else:
        beta = 0.0  # costs do not contribute to policy optimization
        print('Not using costs')

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    if proc_id()==0:
        var_counts = tuple(count_vars(scope) for scope in 
                        ['main/pi', 'main/qr1', 'main/qr2', 'main/qc', 'main'])
        print(('\nNumber of parameters: \t pi: %d, \t qr1: %d, \t qr2: %d, \t qc: %d, \t total: %d\n')%var_counts)

    cost_normalizer = 1
    if use_costs:
        if cost_constraint is None:
            # Convert assuming equal cost accumulated each step
            # Note this isn't the case, since the early in episode doesn't usually have cost,
            # but since our algorithm optimizes the discounted infinite horizon from each entry
            # in the replay buffer, we should be approximately correct here.
            # It's worth checking empirical total undiscounted costs to see if they match.
            if np.abs(safety_budget) >= 1e-6:
                cost_constraint = 1
                cost_normalizer = np.abs(safety_budget)
                if 0 <= safety_discount_factor < 1.0:
                    cost_normalizer *= (1 - safety_discount_factor ** max_ep_len) / (1 - safety_discount_factor) / max_ep_len
            else:
                cost_normalizer = 1
                cost_constraint = safety_budget
                if 0 <= safety_discount_factor < 1.0:
                    cost_constraint *= (1 - safety_discount_factor ** max_ep_len) / (1 - safety_discount_factor) / max_ep_len
        print('using cost constraint', cost_constraint)        
    # Min Double-Q:
    min_q_pi = tf.minimum(qr1_pi, qr2_pi)
    min_q_pi_targ = tf.minimum(qr1_pi_targ, qr2_pi_targ)

    # Targets for Q and V regression with normalized costs
    q_backup = tf.stop_gradient(r_ph   + discount_factor*(1-d_ph)*(min_q_pi_targ - alpha * logp_pi2))
    qc_backup = tf.stop_gradient(c_ph / cost_normalizer + safety_discount_factor*(1-d_ph)*qc_pi_targ)
    damp = 0
    if use_costs:
        violation = tf.reduce_mean(cost_constraint - qc)     # normalized costs         
    

    # Soft actor-critic losses
    qr1_loss = 0.5 * tf.reduce_mean((q_backup - qr1)**2)
    qr2_loss = 0.5 * tf.reduce_mean((q_backup - qr2)**2)
    qc_loss = 0.5 * tf.reduce_mean((qc_backup - qc)**2)
    q_loss = qr1_loss + qr2_loss + qc_loss
    pi_loss = tf.reduce_mean(alpha * logp_pi - min_q_pi + (beta - damp) * qc_pi)

    # Loss for alpha
    target_entropy *= act_dim
    pi_entropy = -tf.reduce_mean(logp_pi)
    # alpha_loss = - soft_alpha * (entropy_constraint - pi_entropy)
    alpha_loss = - alpha * (target_entropy - pi_entropy)
    print('using entropy constraint', target_entropy)

    # Loss for beta
    if use_costs:
        beta_loss = beta * (cost_constraint - qc)   # normalized costs 
    
    # Policy train op
    # (has to be separate from value train op, because qr1_pi appears in pi_loss)
    train_pi_op = MpiAdamOptimizer(learning_rate=lr).minimize(pi_loss, var_list=get_vars('main/pi'), name='train_pi')

    # Value train op
    with tf.control_dependencies([train_pi_op]):
        train_q_op = MpiAdamOptimizer(learning_rate=lr).minimize(q_loss, var_list=get_vars('main/q'), name='train_q')

    if fixed_entropy_bonus is None:
        entreg_optimizer = MpiAdamOptimizer(learning_rate=alpha_lr)
        with tf.control_dependencies([train_q_op]):
            train_entreg_op = entreg_optimizer.minimize(alpha_loss, var_list=get_vars('entreg'))

    if use_costs and fixed_cost_penalty is None:
        costpen_optimizer = MpiAdamOptimizer(learning_rate=penalty_lr)
        with tf.control_dependencies([train_entreg_op]):
            train_costpen_op = costpen_optimizer.minimize(beta_loss, var_list=get_vars('costpen'))

    # Polyak averaging for target variables
    target_update = get_target_update('main', 'target', polyak)

    # Single monolithic update with explicit control dependencies
    with tf.control_dependencies([train_pi_op]):
        with tf.control_dependencies([train_q_op]):
            grouped_update = tf.group([target_update])

    if fixed_entropy_bonus is None:
        grouped_update = tf.group([grouped_update, train_entreg_op])
    if use_costs and fixed_cost_penalty is None:
        grouped_update = tf.group([grouped_update, train_costpen_op])

    # Initializing targets to match main variables
    # As a shortcut, use our exponential moving average update w/ coefficient zero
    target_init = get_target_update('main', 'target', 0.0)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)
    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph},
                                outputs={'mu': mu, 'pi': pi, 'qr1': qr1, 'qr2': qr2, 'qc': qc})

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]

    def test_agent(n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_cost, ep_len, ep_goals, = test_env.reset(), 0, False, 0, 0, 0, 0
            if saute_constraints:
                true_ep_ret = 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, info = test_env.step(get_action(o, True))
                if render and proc_id() == 0 and j == 0:
                    test_env.render()
                ep_ret += r
                ep_cost += info.get('cost', 0)
                ep_len += 1
                ep_goals += 1 if info.get('goal_met', False) else 0
                if saute_constraints:
                    true_ep_ret += info['true_reward']
            if saute_constraints:
                logger.store(TestEpRet=true_ep_ret, TestEpCost=ep_cost, TestEpLen=ep_len, TestEpGoals=ep_goals)
            else:
                logger.store(TestEpRet=ep_ret, TestEpCost=ep_cost, TestEpLen=ep_len, TestEpGoals=ep_goals)

    start_time = time.time()
    o, r, d, ep_ret, ep_cost, ep_len, ep_goals = env.reset(), 0, False, 0, 0, 0, 0
    if saute_constraints:
        true_ep_ret = 0
    total_steps = steps_per_epoch * epochs

    # variables to measure in an update
    vars_to_get = dict(LossPi=pi_loss, LossQR1=qr1_loss, LossQR2=qr2_loss, LossQC=qc_loss,
                       QR1Vals=qr1, QR2Vals=qr2, QCVals=qc, LogPi=logp_pi, PiEntropy=pi_entropy,
                       Alpha=alpha, LogAlpha=log_alpha, LossAlpha=alpha_loss)
    if use_costs:
        vars_to_get.update(dict(Beta=beta, LogBeta=log_beta, LossBeta=beta_loss, Violation=violation))

    print('starting training', proc_id())

    # Main loop: collect experience in env and update/log each epoch
    local_steps = 0    
    cum_cost = 0
    local_steps_per_epoch = steps_per_epoch // num_procs()
    local_batch_size = batch_size // num_procs()
    epoch_start_time = time.time()
    df = pd.DataFrame()
    assert max_ep_len <= local_steps_per_epoch, "Episode length should be smaller or equal to local steps per epoch"

    training_rewards = deque([0], maxlen= steps_per_epoch // max_ep_len)
    training_costs = deque([0], maxlen= steps_per_epoch // max_ep_len) 

    for t in range(total_steps // num_procs()):
        """
        Until local_start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards,
        use the learned policy.
        """
        if t > local_start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, info = env.step(a)
        r *= reward_scale  # yee-haw
        c = info.get('cost', 0)

         # Track cumulative cost over training
        cum_cost += c

        ep_ret += r
        ep_cost += c
        ep_len += 1
        ep_goals += 1 if info.get('goal_met', False) else 0
        local_steps += 1
        if saute_constraints:
            true_ep_ret += info['true_reward']
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d, c)
        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        if d or (ep_len == max_ep_len):
            if saute_constraints:
                logger.store(EpRet=true_ep_ret, EpCost=ep_cost, EpLen=ep_len, EpGoals=ep_goals)
                training_rewards.extend([true_ep_ret])
                true_ep_ret = 0
            else:                
                logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len, EpGoals=ep_goals)
                training_rewards.extend([ep_ret])
            training_costs.extend([ep_cost])  
            o, r, d, ep_ret, ep_cost, ep_len, ep_goals = env.reset(), 0, False, 0, 0, 0, 0

        if t > 0 and t % train_frequency == 0:
            for j in range(train_frequency):
                batch = replay_buffer.sample_batch(local_batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             c_ph: batch['costs'],
                             d_ph: batch['done'],
                            }
                if t < local_update_after:
                    logger.store(**sess.run(vars_to_get, feed_dict))
                else:
                    values, _ = sess.run([vars_to_get, grouped_update], feed_dict)
                    logger.store(**values)


        # End of epoch wrap-up
        if t > 0 and t % local_steps_per_epoch == 0:
            epoch = t // local_steps_per_epoch

            #=====================================================================#
            #  Cumulative cost calculations                                       #
            #=====================================================================#

            cumulative_cost = mpi_sum(cum_cost)
            cost_rate = cumulative_cost / ((epoch+1)*steps_per_epoch) * max_ep_len # cost rate per episode

            # Save model
            if (checkpoint_frequency and (epoch % checkpoint_frequency == 0)) or (epoch == epochs-1):
                logger.save_state({'env': env}, epoch)
            if  len(training_rewards) == 1:
                cur_df = pd.DataFrame({
                            "episode_return": training_rewards[0], 
                            "episode_cost": training_costs[0], 
                            "accumulated_cost": cumulative_cost,
                            "cost_rate": cost_rate,
                            "epoch": epoch,
                            "run": 0
                        }, index=[epoch])
            else:
                cur_df = pd.DataFrame({
                            "episode_return": training_rewards, 
                            "episode_cost": training_costs, 
                            "accumulated_cost": cumulative_cost,
                            "cost_rate": cost_rate,
                            "epoch": epoch,
                            "run": np.arange(len(training_rewards))
                        })
            df = df.append(cur_df)            
            df.to_csv(os.path.join(logger.output_dir, "train_results.csv"))

            # Test the performance of the deterministic version of the agent.
            test_start_time = time.time()
            if test_frequency and (epoch % test_frequency == 0 or epoch == 1):                
                test_agent(n=n_test_episodes) # minimal testing during training 
            logger.store(TestTime=time.time() - test_start_time)

            logger.store(EpochTime=time.time() - epoch_start_time)
            epoch_start_time = time.time()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpCost', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('EpGoals', average_only=True)
            if test_frequency and (epoch % test_frequency == 0 or epoch == 1):   
                logger.log_tabular('TestEpRet', with_min_and_max=True)
                logger.log_tabular('TestEpCost', with_min_and_max=True)
                logger.log_tabular('TestEpLen', average_only=True)
                logger.log_tabular('TestEpGoals', average_only=True)
            logger.log_tabular('TotalEnvInteracts', mpi_sum(local_steps))
            logger.log_tabular('QR1Vals', with_min_and_max=True)
            logger.log_tabular('QR2Vals', with_min_and_max=True)
            logger.log_tabular('QCVals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQR1', average_only=True)
            logger.log_tabular('LossQR2', average_only=True)
            logger.log_tabular('LossQC', average_only=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('LogAlpha', average_only=True)
            logger.log_tabular('Alpha', average_only=True)
            logger.log_tabular('CostRate', cost_rate)
            logger.log_tabular('CumulativeCost', cumulative_cost)

            if use_costs:
                logger.log_tabular('LossBeta', average_only=True)
                logger.log_tabular('LogBeta', average_only=True)
                logger.log_tabular('Beta', average_only=True)
                logger.log_tabular('Violation', average_only=True)


            logger.log_tabular('PiEntropy', average_only=True)
            # logger.log_tabular('TestTime', average_only=True)
            logger.log_tabular('EpochTime', average_only=True)
            logger.log_tabular('TotalTime', time.time()-start_time)
            if writer is not None:
                # optimization infos
                writer.add_scalar('train_info/LossPi', logger.log_current_row['LossPi'], epoch)
                writer.add_scalar('train_info/LossQC', logger.log_current_row['LossQC'], epoch)
                writer.add_scalar('train_info/LossQR1', logger.log_current_row['LossQR1'], epoch)
                writer.add_scalar('train_info/LossQR2', logger.log_current_row['LossQR2'], epoch)
                writer.add_scalar('train_info/LossAlpha', logger.log_current_row['LossAlpha'], epoch)
                writer.add_scalar('train_info/Alpha', logger.log_current_row['Alpha'], epoch)
                writer.add_scalar('train_info/std_QR1Vals', logger.log_current_row['StdQR1Vals'], epoch)
                writer.add_scalar('train_info/mean_QR1Vals', logger.log_current_row['AverageQR1Vals'], epoch)
                writer.add_scalar('train_info/std_QR2Vals', logger.log_current_row['StdQR2Vals'], epoch)
                writer.add_scalar('train_info/mean_QR2Vals', logger.log_current_row['AverageQR2Vals'], epoch)
                if use_costs:
                    writer.add_scalar('train_info/std_QCVals', logger.log_current_row['StdQCVals'], epoch)
                    writer.add_scalar('train_info/mean_QCVals', logger.log_current_row['AverageQCVals'], epoch)
                    writer.add_scalar('train_info/LossBeta', logger.log_current_row['LossBeta'], epoch)
                    writer.add_scalar('train_info/Beta', logger.log_current_row['Beta'], epoch)
 
                # training costs
                # episode return 
                writer.add_scalar('train_return/StdEpRet', logger.log_current_row['StdEpRet'], epoch)
                writer.add_scalar('train_return/AverageEpRet', logger.log_current_row['AverageEpRet'], epoch)
                writer.add_scalar('train_return/MaxEpRet', logger.log_current_row['MaxEpRet'], epoch)
                writer.add_scalar('train_return/MinEpRet', logger.log_current_row['MinEpRet'], epoch)
                # accumulative cost
                writer.add_scalar('train_acc_cost/CumulativeCost', logger.log_current_row['CumulativeCost'], epoch)
                writer.add_scalar('train_acc_cost/CostRate', logger.log_current_row['CostRate'], epoch)
                # episode cost 
                if use_costs:
                    writer.add_scalar('train_cost/violation', logger.log_current_row['Violation'], epoch)
                writer.add_scalar('train_cost/StdEpCost', logger.log_current_row['StdEpCost'], epoch)
                writer.add_scalar('train_cost/AverageEpCost', logger.log_current_row['AverageEpCost'], epoch)
                writer.add_scalar('train_cost/MaxEpCost', logger.log_current_row['MaxEpCost'], epoch)
                writer.add_scalar('train_cost/MinEpCost', logger.log_current_row['MinEpCost'], epoch)
                if test_frequency and (epoch % test_frequency == 0 or epoch == 1):   
                    # testing costs
                    # episode return 
                    writer.add_scalar('test_return/StdEpRet', logger.log_current_row['StdTestEpRet'], epoch)
                    writer.add_scalar('test_return/AverageEpRet', logger.log_current_row['AverageTestEpRet'], epoch)
                    writer.add_scalar('test_return/MaxEpRet', logger.log_current_row['MaxTestEpRet'], epoch)
                    writer.add_scalar('test_return/MinEpRet', logger.log_current_row['MinTestEpRet'], epoch)
                    # episode cost 
                    writer.add_scalar('test_cost/StdEpCost', logger.log_current_row['StdTestEpCost'], epoch)
                    writer.add_scalar('test_cost/AverageEpCost', logger.log_current_row['AverageTestEpCost'], epoch)
                    writer.add_scalar('test_cost/MaxEpCost', logger.log_current_row['MaxTestEpCost'], epoch)
                    writer.add_scalar('test_cost/MinEpCost', logger.log_current_row['MinTestEpCost'], epoch)

            logger.dump_tabular()
    sess.close() 
    tf.reset_default_graph()


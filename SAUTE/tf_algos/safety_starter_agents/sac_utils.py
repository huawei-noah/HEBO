#!/usr/bin/env python
"""
Copied from safety starter agents.
"""
import numpy as np
import tensorflow as tf

EPS = 1e-8

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def get_target_update(main_name, target_name, polyak):
    ''' Get a tensorflow op to update target variables based on main variables '''
    main_vars = {x.name: x for x in get_vars(main_name)}
    targ_vars = {x.name: x for x in get_vars(target_name)}
    assign_ops = []
    for v_targ in targ_vars:
        assert v_targ.startswith(target_name), f'bad var name {v_targ} for {target_name}'
        v_main = v_targ.replace(target_name, main_name, 1)
        assert v_main in main_vars, f'missing var name {v_main}'
        assign_op = tf.assign(targ_vars[v_targ], polyak*targ_vars[v_targ] + (1-polyak)*main_vars[v_main])
        assign_ops.append(assign_op)
    return tf.group(assign_ops)


"""
Policies
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation):
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes), activation, activation)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)
    log_std = tf.layers.dense(net, act_dim, activation=None)
    log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return mu, pi, logp_pi

def apply_squashing_func(mu, pi, logp_pi):
    # Adjustment to log prob
    logp_pi -= tf.reduce_sum(2*(np.log(2) - pi - tf.nn.softplus(-2*pi)), axis=1)

    # Squash those unbounded actions!
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    return mu, pi, logp_pi


"""
Actors and Critics
"""
def mlp_actor(x, a, name='pi', hidden_sizes=(256,256), activation=tf.nn.relu,
              output_activation=None, policy=mlp_gaussian_policy, action_space=None):
    # policy
    with tf.variable_scope(name):
        mu, pi, logp_pi = policy(x, a, hidden_sizes, activation, output_activation)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

    # make sure actions are in correct range
    action_scale = action_space.high[0]
    mu *= action_scale
    pi *= action_scale

    return mu, pi, logp_pi


def mlp_critic(x, a, pi, name, hidden_sizes=(256,256), activation=tf.nn.relu,
               output_activation=None, policy=mlp_gaussian_policy, action_space=None):

    fn_mlp = lambda x : tf.squeeze(mlp(x=x,
                                       hidden_sizes=list(hidden_sizes)+[1],
                                       activation=activation,
                                       output_activation=None),
                                   axis=1)
    with tf.variable_scope(name):
        critic = fn_mlp(tf.concat([x,a], axis=-1))

    with tf.variable_scope(name, reuse=True):
        critic_pi = fn_mlp(tf.concat([x,pi], axis=-1))

    return critic, critic_pi

def mlp_var(x, a, pi, name, hidden_sizes=(64,64), activation=tf.nn.relu,
              output_activation=None, policy=mlp_gaussian_policy, action_space=None):
    
    fn_mlp = lambda x : tf.squeeze(mlp(x=x,
                                       hidden_sizes=list(hidden_sizes)+[1],
                                       activation=activation,
                                       output_activation=None),
                                   axis=1)
    
    with tf.variable_scope(name):
        var = fn_mlp(tf.concat([x,a], axis=-1))
        var = tf.nn.softplus(var)

    with tf.variable_scope(name, reuse=True):
        var_pi = fn_mlp(tf.concat([x,pi], axis=-1))
        var_pi = tf.nn.softplus(var_pi)

    return var, var_pi

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.costs_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, cost):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.costs_buf[self.ptr] = cost
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    costs=self.costs_buf[idxs],
                    done=self.done_buf[idxs])


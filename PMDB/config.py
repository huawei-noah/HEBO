import argparse


def args_setting():
    parser = argparse.ArgumentParser(description='Args')

    # Seed
    parser.add_argument('--seed', type=int, default=0)

    # Task:
    parser.add_argument('--task', default="hopper-medium-v2",
                        help='D4RL environment (default: hopper-medium-v2)')

    # Dynamics Model:
    parser.add_argument('--dynamics_path', default=None,
                        help='path to load saved dynamics model (default: None)')
    parser.add_argument('--dynamics_save_path', default='model_rep',
                        help='path to save dynamics model (default: None)')

    parser.add_argument('--ensemble_size', type=int, default=100,
                        help='size of dynamics ensemble (default: 100)')
    parser.add_argument('--transition_layer_size', type=int, default=256,
                        help='hidden size per layer of dynamics model (default: 256)')
    parser.add_argument('--transition_layers', type=int, default=4,
                        help='number of hidden layers of dynamics model (default: 4)')

    parser.add_argument('--transition_num_epoch', type=int, default=1000,
                        help='number of epochs to train dynamics model (default: 1000)')
    parser.add_argument('--transition_batch_size', default=256,
                        help='batch size of dynamics loss (default: 256)')
    parser.add_argument('--transition_lr', type=float, default=1e-4,
                        help='dynamics learning rate (default: 1e-4)')

    # Agent:
    parser.add_argument('--policy_type', default="Gaussian",
                        help='policy type: Gaussian | Real_NVP (default: Gaussian)')
    parser.add_argument('--det_policy', action="store_true",
                        help='deterministic policy, valid for policy_type=Gaussian (default: False)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau_value', type=float, default=5e-3,
                        help='smoothing coefficient for value target (default: 5e-3)')
    parser.add_argument('--tau_policy', type=float, default=1e-5,
                        help='smoothing coefficient for policy target (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='weight α to determine the relative importance of the KL\
                              term against the reward (default: 0.1)')
    parser.add_argument('--automatic_alpha_tuning', action="store_false",
                        help='automaically adjust α (default: True)')
    parser.add_argument('--target_kld', type=float, default=5.,
                        help='target KL divergence when automatically adjusting α (default: 5.)')

    parser.add_argument('--agent_num_steps', type=int, default=int(2e6),
                        help='number of gradient steps for policy learning (default: 2e6)')
    parser.add_argument('--real_batch_size', type=int, default=128,
                        help='batch size of MDP loss (default: 128)')
    parser.add_argument('--adv_batch_size', type=int, default=128,
                        help='batch size of AMG loss (default: 128)')

    parser.add_argument('--actor_lr', type=float, default=3e-5,
                        help='actor learning rate (default: 3e-5)')
    parser.add_argument('--critic_lr', type=float, default=3e-4,
                        help='critic learning rate (default: 3e-4)')

    parser.add_argument('--agent_layer_size', type=int, default=256,
                        help='hidden size of actor and critic (default: 256)')

    parser.add_argument('--explore_ratio', type=float, default=0.,
                        help='exploration probability of primary player (default:0.)')

    # Adversarial Dynamics:
    parser.add_argument('--adv_horizon', type=int, default=1000,
                        help='AMG horizon (default: 1000)')
    parser.add_argument('--adv_explore_ratio', type=float, default=0.1,
                        help='exploration probability of adversarial player (default: 0.1)')
    parser.add_argument('--num_sample_transition', type=int, default=10,
                        help='hyperparameter N for dynamics sampling (default: 10)')
    parser.add_argument('--order_transition', type=int, default=2,
                        help='hyperparameter k for dynamics sampling (default: 2)')
    parser.add_argument('--MC_size_state', type=int, default=10,
                        help='state sample size for estimating expectation (default: 10)')
    parser.add_argument('--MC_size_action', type=int, default=20,
                        help='action sample size for estimating expectation (default: 20)')

    # Common:
    parser.add_argument('--cpu', action="store_true",
                        help='run on CPU (default: False)')
    parser.add_argument('--eval', action="store_false",
                        help='periodically evaluate on real environment (default: True)')

    return parser.parse_args()

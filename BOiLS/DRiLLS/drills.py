#!/usr/bin/python3

# Copyright (c) 2019, SCALE Lab, Brown University
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

# 2021.11.10-Add paramaters to control the runs (type of reference, seeds...)
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import argparse
import os
import sys
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path[0] = ROOT_PROJECT
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from DRiLLS.utils import _filter_kwargs
from DRiLLS.drills.models.gym_agents import AgentA2C, AgentPPO2
from utils.utils_cmd import parse_list
from DRiLLS.drills.exps.exp_gym import ExpDQN, ExpOnPolicy, ExpPPO
from DRiLLS.drills.exps.exp_tf import ExpTF

class CapitalisedHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        if prefix is None:
            prefix = 'Usage: '
            return super(CapitalisedHelpFormatter, self).add_usage(usage, actions, groups, prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True, formatter_class=CapitalisedHelpFormatter,
                                     description='Performs logic synthesis optimization using RL')
    parser.register('type', list, parse_list)
    parser._positionals.title = 'Positional arguments'
    parser._optionals.title = 'Optional arguments'
    parser.add_argument('-v', '--version', action='version',
                        version='DRiLLS v0.1', help="Shows program's version number and exit")
    parser.add_argument("-s", "--fixed_script", type=open,
                        help="Executes a fixed optimization script before DRiLLS")
    parser.add_argument("-e", "--episodes", type=int, default=50, help="Number of training episodes to run")
    parser.add_argument("-i", "--iterations", type=int, default=50, help="Number of iterations to run for each episode")
    parser.add_argument("-d", "--design_id", required=True, type=str,
                        help=' design file id (will load a .blif)')
    parser.add_argument("-a", "--agent", required=True, type=str, help="Agent type ('tf-A2C', 'PPO'...)")
    parser.add_argument("--mode", type=str, choices=['train', 'optimize'],
                        help="Use the design to train the model or only optimize it")

    parser.add_argument("--test_designs", type=list, default=None,
                        help="If mode is optimized, list of design files in one of the accepted formats by ABC")

    parser.add_argument("--mapping", type=str, choices=['scl', 'fpga'], default='fpga',
                        help="Map to standard cell library or FPGA")
    parser.add_argument("--lut_inputs", type=int, required=True, help="number of LUT inputs (2 < num < 33)")
    parser.add_argument("--action_space_id", type=str, default='standard',
                        help="id of action space defining avaible abc optimisation operations")
    parser.add_argument("--objective", type=str, choices=('lut', 'both', 'level', 'min_improvements'), required=True,
                        help="which objective should be optimized")
    parser.add_argument("--seed", nargs='+', type=int, required=True, help="Seed for reproducibility")

    parser.add_argument("--rec", action='store_true', help="Whether to use a recurrent policy")
    parser.add_argument("--n_lstm", type=int, default=32, help="The number of LSTM cells (for recurrent policies)")
    parser.add_argument("--layer_norm", action='store_true', help="Whether to use layer normalisation")
    parser.add_argument("--pi_arch", type=list, default=[20, 20], help="Actor net architecture")
    parser.add_argument("--vf_arch", type=list, default=[10], help="Value function net architecture")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient for the loss calculation")
    parser.add_argument("--lr", type=float, default=0.005, help="Value function net architecture")
    parser.add_argument("--feature_extraction", type=str, default='mlp', choices=['mlp', 'cnn'],
                        help=" The feature extraction type (“cnn” or “mlp”)")
    # DQN
    parser.add_argument("--dueling", action='store_true',
                        help="double the output MLP to compute a baseline for action scores")
    parser.add_argument("--double_q", action='store_true', help="Whether to enable Double-Q learning or not.")

    parser.add_argument("--overwrite", action='store_true', help="Whether to overwrite")

    # greedy
    parser.add_argument("--k", default=6, type=int, help="Parameter of the lazy greedy search")
    parser.add_argument("--ref_abc_seq", type=str, default='resyn2', help="ID of reference sequence used to "
                                                                          "measure improvements on area and delay")
    parser.add_argument("--metric", default='constraint', type=str, help="Aggregation of improvement on delay and area "
                                                                         "considered (min, sum, constraint)")

    args_ = parser.parse_args()

    for seed in args_.seed:
        if args_.mode == 'optimize':
            assert args_.test_designs is not None and len(args_.test_designs) > 0, args_.test_designs

        exp_kw = dict(
            max_iteration=args_.iterations,
            design_id=args_.design_id,
            load_model=None,
            episodes=args_.episodes,
            mapping=args_.mapping,
            lut_inputs=args_.lut_inputs,
            action_space_id=args_.action_space_id,
            seed=seed,
            objective=args_.objective,

            # greedy
            k=args_.k,
            ref_abc_seq=args_.ref_abc_seq,
            metric=args_.metric,

            # rl
            rec=args_.rec,
            n_lstm=args_.n_lstm,
            pi_arch=args_.pi_arch,
            vf_arch=args_.vf_arch,
            ent_coef=args_.ent_coef,
            layer_norm=args_.layer_norm,
            feature_extraction=args_.feature_extraction,

            double_q=args_.double_q,
            dueling=args_.dueling,

            learning_rate=args_.lr
        )

        if args_.rec and args_.agent not in ['ppo2', 'a2c']:
            raise ValueError('Should not activate rec option for ExpRandom')
        if args_.agent == 'tf-A2C':
            exp: ExpTF = ExpTF(**_filter_kwargs(ExpTF, **exp_kw))
        elif args_.agent == 'ppo':
            exp: ExpPPO = ExpPPO(**exp_kw)
        elif args_.agent == 'ppo2':
            exp: ExpOnPolicy = ExpOnPolicy(agent_class=AgentPPO2, **_filter_kwargs(ExpOnPolicy, **exp_kw))
        elif args_.agent == 'dqn':
            exp: ExpDQN = ExpDQN(**_filter_kwargs(ExpDQN, **exp_kw))
        elif args_.agent == 'a2c':
            exp: ExpOnPolicy = ExpOnPolicy(agent_class=AgentA2C, **_filter_kwargs(ExpOnPolicy, **exp_kw))
        else:
            raise ValueError(args_.agent)

        if args_.fixed_script:
            raise NotImplementedError()
            # args.params = optimize_with_fixed_script(args.params, args.fixed_script)

        if args_.mode == 'train':
            if exp.already_trained_() and not args_.overwrite:
                print(f"Experiment already trained: stored in {exp.playground_dir}")
            else:
                exp.train()
        elif args_.mode == 'optimize':
            assert exp.already_trained_(), 'Need a trained model'
            exp.optimize(params_file=exp_kw['params_file'], design_files=args_.test_designs,
                         max_iterations=args_.iterations,
                         overwrite=args_.overwrite)
        else:
            raise ValueError(f"args mode should be in ['train', 'optimize'], got: {args_.mode}")

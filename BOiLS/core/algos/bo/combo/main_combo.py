# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved. Redistribution and use in source and binary 
# forms, with or without modification, are permitted provided that the following conditions are met: 
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer. 
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the 
# following disclaimer in the documentation and/or other materials provided with the distribution. 
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote 
# products derived from this software without specific prior written permission. 
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE 
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent.parent)
sys.path[0] = ROOT_PROJECT
import torch

from utils.utils_cmd import parse_list
from core.algos.bo.combo.combo_exp import COMBOExp
from resources.COMBO import DiffusionKernel
from resources.COMBO.graphGP.models.gp_regression import GPRegression
from resources.COMBO.graphGP.sampler.sample_posterior import posterior_sampling

from resources.COMBO import next_evaluation
from resources.COMBO import expected_improvement
from resources.COMBO.acquisition.acquisition_marginalization import inference_sampling

from resources.COMBO.utils import displaying_and_logging


def run_suggest(surrogate_model, eval_inputs, eval_outputs, n_vertices, adj_mat_list, log_beta, sorted_partition,
                acquisition_func, parallel):
    start_time = time.time()
    reference = torch.min(eval_outputs, dim=0)[0].item()
    # print('(%s) Sampling' % time.strftime('%H:%M:%S', time.gmtime()))
    sample_posterior = posterior_sampling(surrogate_model, eval_inputs, eval_outputs, n_vertices, adj_mat_list,
                                          log_beta, sorted_partition, n_sample=10, n_burn=0, n_thin=1)
    hyper_samples, log_beta_samples, partition_samples, freq_samples, basis_samples, edge_mat_samples = sample_posterior
    log_beta = log_beta_samples[-1]
    sorted_partition = partition_samples[-1]

    x_opt = eval_inputs[torch.argmin(eval_outputs)]
    inference_samples = inference_sampling(eval_inputs, eval_outputs, n_vertices,
                                           hyper_samples, log_beta_samples, partition_samples,
                                           freq_samples, basis_samples)
    suggestion = next_evaluation(x_opt, eval_inputs, inference_samples, partition_samples, edge_mat_samples,
                                 n_vertices, acquisition_func, reference, parallel)
    processing_time = time.time() - start_time
    return suggestion, log_beta, sorted_partition, processing_time


def run_bo(exp_dirname, task, objective: COMBOExp, store_data, parallel):
    bo_data_filename = os.path.join(exp_dirname, 'bo_data.pt')
    bo_data = torch.load(bo_data_filename)
    surrogate_model = bo_data['surrogate_model']
    eval_inputs = bo_data['eval_inputs']
    eval_outputs = bo_data['eval_outputs']
    n_vertices = bo_data['n_vertices']
    adj_mat_list = bo_data['adj_mat_list']
    log_beta = bo_data['log_beta']
    sorted_partition = bo_data['sorted_partition']
    time_list = bo_data['time_list']
    elapse_list = bo_data['elapse_list']
    pred_mean_list = bo_data['pred_mean_list']
    pred_std_list = bo_data['pred_std_list']
    pred_var_list = bo_data['pred_var_list']
    acquisition_func = bo_data['acquisition_func']

    updated = False

    if eval_inputs.size(0) == eval_outputs.size(0) and task in ['suggest', 'both']:
        suggestion, log_beta, sorted_partition, processing_time = run_suggest(
            surrogate_model=surrogate_model, eval_inputs=eval_inputs, eval_outputs=eval_outputs, n_vertices=n_vertices,
            adj_mat_list=adj_mat_list, log_beta=log_beta, sorted_partition=sorted_partition,
            acquisition_func=acquisition_func, parallel=parallel)

        next_input, pred_mean, pred_std, pred_var = suggestion
        eval_inputs = torch.cat([eval_inputs, next_input.view(1, -1)], 0)
        elapse_list.append(processing_time)
        pred_mean_list.append(pred_mean.item())
        pred_std_list.append(pred_std.item())
        pred_var_list.append(pred_var.item())

        updated = True

    if eval_inputs.size(0) - 1 == eval_outputs.size(0) and task in ['evaluate', 'both']:
        next_output = objective.evaluate(eval_inputs[-1]).view(1, 1)
        eval_outputs = torch.cat([eval_outputs, next_output])
        assert not torch.isnan(eval_outputs).any()

        time_list.append(time.time())

        updated = True

    if updated:
        bo_data = {'surrogate_model': surrogate_model, 'eval_inputs': eval_inputs, 'eval_outputs': eval_outputs,
                   'n_vertices': n_vertices, 'adj_mat_list': adj_mat_list, 'log_beta': log_beta,
                   'sorted_partition': sorted_partition, 'acquisition_func': acquisition_func,
                   'time_list': time_list, 'elapse_list': elapse_list,
                   'pred_mean_list': pred_mean_list, 'pred_std_list': pred_std_list, 'pred_var_list': pred_var_list}
        torch.save(bo_data, bo_data_filename)

        displaying_and_logging(os.path.join(exp_dirname, 'log'), eval_inputs, eval_outputs,
                               pred_mean_list, pred_std_list, pred_var_list,
                               time_list, elapse_list, store_data)

    return eval_outputs.size(0)


def COMBO(objective: COMBOExp, n_eval: int = 200, parallel=False, store_data=False, task='both', **kwargs):
    """

    Args:
        objective:
        n_eval:
        parallel:
        store_data:
        task:
        **kwargs:

    Returns:

    """
    assert task in ['suggest', 'evaluate', 'both']
    # GOLD continues from info given in 'path' or starts minimization of 'objective'
    acquisition_func = expected_improvement

    exp_dirname = objective.exp_path()

    n_vertices = objective.n_vertices
    adj_mat_list = objective.adjacency_mat
    grouped_log_beta = torch.ones(len(objective.fourier_freq))
    fourier_freq_list = objective.fourier_freq
    fourier_basis_list = objective.fourier_basis
    suggested_init = objective.suggested_init  # suggested_init should be 2d tensor
    n_init = suggested_init.size(0)

    kernel = DiffusionKernel(grouped_log_beta=grouped_log_beta,
                             fourier_freq_list=fourier_freq_list, fourier_basis_list=fourier_basis_list)
    surrogate_model = GPRegression(kernel=kernel)

    eval_inputs = suggested_init
    eval_outputs = torch.zeros(eval_inputs.size(0), 1, device=eval_inputs.device)
    for i in range(eval_inputs.size(0)):
        eval_outputs[i] = objective.evaluate(eval_inputs[i])
    assert not torch.isnan(eval_outputs).any()
    log_beta = eval_outputs.new_zeros(eval_inputs.size(1))
    sorted_partition = [[m] for m in range(eval_inputs.size(1))]

    time_list = [time.time()] * n_init
    elapse_list = [0] * n_init
    pred_mean_list = [0] * n_init
    pred_std_list = [0] * n_init
    pred_var_list = [0] * n_init

    surrogate_model.init_param(eval_outputs)
    objective.log('(%s) Burn-in' % time.strftime('%H:%M:%S', time.gmtime()))
    sample_posterior = posterior_sampling(surrogate_model, eval_inputs, eval_outputs, n_vertices, adj_mat_list,
                                          log_beta, sorted_partition, n_sample=1, n_burn=99, n_thin=1)
    log_beta = sample_posterior[1][0]
    sorted_partition = sample_posterior[2][0]

    bo_data = {'surrogate_model': surrogate_model, 'eval_inputs': eval_inputs, 'eval_outputs': eval_outputs,
               'n_vertices': n_vertices, 'adj_mat_list': adj_mat_list, 'log_beta': log_beta,
               'sorted_partition': sorted_partition, 'time_list': time_list, 'elapse_list': elapse_list,
               'pred_mean_list': pred_mean_list, 'pred_std_list': pred_std_list, 'pred_var_list': pred_var_list,
               'acquisition_func': acquisition_func}
    torch.save(bo_data, os.path.join(exp_dirname, 'bo_data.pt'))

    eval_cnt = 0
    while eval_cnt < n_eval:
        if (eval_cnt + 1) % 10 == 0:
            objective.log(f"{eval_cnt} / {n_eval}")
        eval_cnt = run_bo(exp_dirname=exp_dirname,
                          objective=objective, store_data=store_data, task=task, parallel=parallel)


if __name__ == '__main__':
    parser_ = argparse.ArgumentParser(
        description='COMBO : Combinatorial Bayesian Optimization using the graph Cartesian product')
    parser_.register('type', list, parse_list)

    parser_.add_argument('--n_eval', required=True, type=int)
    parser_.add_argument('--n_initial', type=int, default=20, help="Number of initial random evaluations")
    parser_.add_argument('--lamda', type=float, default=None)
    parser_.add_argument('--parallel', dest='parallel', action='store_true', default=False)
    parser_.add_argument('--device', dest='device', type=int, default=None)
    parser_.add_argument('--task', dest='task', type=str, default='both')

    # EDA
    parser_.add_argument("--design_file", type=str, required=True, help="Design file")
    parser_.add_argument("--design_files", type=list, default=[], help="List of design files")
    parser_.add_argument("--seq_length", type=int, required=True, help="length of the optimal sequence to find")
    parser_.add_argument("--mapping", type=str, default='fpga', choices=('fpga', 'scl'),
                         help="Map to standard cell library or FPGA")
    parser_.add_argument("--lut_inputs", type=int, default=4, help="number of LUT inputs (2 < num < 33)")
    parser_.add_argument("--action_space_id", type=str, default='standard',
                         help="id of action space defining avaible abc optimisation operations")
    parser_.add_argument("--library_file", type=str, default=os.path.join(ROOT_PROJECT, 'asap7.lib'),
                         help="library file for mapping")
    parser_.add_argument("--abc_binary", type=str, default='yosys-abc')
    parser_.add_argument("--ref_abc_seq", type=str, default='resyn2',
                         help="sequence of operations to apply to initial design to get reference performance")

    parser_.add_argument("--seed", type=int, default=0, help="seed for reproducibility")

    # common
    parser_.add_argument("--overwrite", action='store_true', help='Overwrite existing experiment')

    args_ = parser_.parse_args()
    kwag_ = vars(args_)
    random_seed_config_ = kwag_['seed']
    parallel_ = kwag_['parallel']
    if args_.device is None:
        del kwag_['device']
    if args_.design_file not in args_.design_files:
        design_files_ = [args_.design_file] + args_.design_files
    else:
        design_files_ = args_.design_files
    for design_file in design_files_:
        exp = COMBOExp(
            design_file=design_file,
            seq_length=args_.seq_length,
            mapping=args_.mapping,
            action_space_id=args_.action_space_id,
            library_file=args_.library_file,
            abc_binary=args_.abc_binary,
            seed=args_.seed,
            lut_inputs=args_.lut_inputs,
            ref_abc_seq=args_.ref_abc_seq,
            n_initial=args_.n_initial,
            lamda=args_.lamda
        )
        overwrite = args_.overwrite

        exist = exp.exists()
        if exist and not overwrite:
            print(f"Experiment already trained: stored in {exp.exp_path()}")
            continue
        elif exist:
            exp.log(f"Overwrite experiment: {exp.exp_path()}")
        result_dir = exp.exp_path()
        exp.log(f'result dir: {result_dir}')
        os.makedirs(result_dir, exist_ok=True)
        logs = ''
        exc: Optional[Exception] = None
        try:
            COMBO(objective=exp, **kwag_)
            res = exp.build_res(verbose=True)
            exp.save_results(res)
        except Exception as e:
            logs = traceback.format_exc()
            exc = e
        f = open(os.path.join(result_dir, 'logs.txt'), "a")
        f.write(logs)
        f.close()
        if exc is not None:
            raise exc

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from TransVAE.transvae.tvae_util import KLAnnealer

# Plotting functions

def plot_test_train_curves(paths, target_path=None, loss_type='tot_loss', data_type='test', labels=None, colors=None):
    """
    Plots the training curves for a set of model log files

    Arguments:
        paths (list, req): List of paths to log files (generated during training)
        target_path (str): Optional path to plot target loss (if you are trying to replicate or improve upon a given loss curve)
        loss_type (str): The type of loss to plot - tot_loss, kld_loss, recon_loss, etc.
        labels (list): List of labels for plot legend
        colors (list): List of colors for each training curve
    """
    if colors is None:
        colors = ['#005073', '#B86953', '#932191', '#90041F', '#0F4935']
    if labels is None:
        labels = []
        for path in paths:
            path = path.split('/')[-1].split('log_GRUGRU_')[-1].split('.')[0]
            labels.append(path)
    plt.figure(figsize=(10,8))
    ax = plt.subplot(111)

    for i, path in enumerate(paths):
        df = pd.read_csv(path)
        try:
            data = df[df.data_type == data_type].groupby('epoch').mean()[loss_type]
        except KeyError:
            data = df[df.data_type == data_type].groupby('epoch').mean()['bce_loss']
        if loss_type == 'kld_loss':
            klannealer = KLAnnealer(1e-8, 0.05, 60, 0)
            klanneal = []
            for j in range(60):
                klanneal.append(klannealer(j))
            data /= klanneal
        plt.plot(data, c=colors[i], lw=2.5, label=labels[i], alpha=0.95)
    if target_path is not None:
        df = pd.read_csv(target_path)
        try:
            target = df[df.data_type == data_type].groupby('epoch').mean()[loss_type]
        except KeyError:
            target = df[df.data_type == data_type].groupby('epoch').mean()['bce_loss']
        plt.plot(target, c='black', ls=':', lw=2.5, label='Approximate Goal')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.yscale('log')
    plt.ylabel(loss_type, rotation='horizontal', labelpad=30)
    plt.xlabel('epoch')
    return plt

def plot_loss_by_type(path, colors=None):
    """
    Plot the training curve of one model for each loss type

    Arguments:
        path (str, req): Path to log file of trained model
        colors (list): Colors for each loss type
    """
    if colors is None:
        colors = ['#005073', '#B86953', '#932191', '#90041F', '#0F4935']

    df = pd.read_csv(path)

    plt.figure(figsize=(10,8))
    ax = plt.subplot(111)

    loss_types = ['tot_loss', 'bce_loss', 'kld_loss', 'pred_loss']
    for i, loss_type in enumerate(loss_types):
        train_data = df[df.data_type == 'train'].groupby('epoch').mean()[loss_type]
        test_data = df[df.data_type == 'test'].groupby('epoch').mean()[loss_type]
        plt.plot(train_data, c=colors[i], label='train_'+loss_type)
        plt.plot(test_data, c=colors[i], label='test_'+loss_type, ls=':')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.yscale('log')
    plt.ylabel('Loss', rotation='horizontal')
    plt.xlabel('epoch')
    plt.title(path.split('/')[-1].split('log_GRUGRU_')[-1].split('.')[0])
    return plt

def plot_reconstruction_accuracies(dir, colors=None):
    """
    Plots token, SMILE and positional reconstruction accuracies for all model types in directory

    Arguments:
        dir (str, req): Directory to json files containing stored accuracies for each trained model
        colors (list): List of colors for each trained model
    """
    if colors is None:
        colors = ['#005073', '#B86953', '#932191', '#90041F', '#0F4935']

    data, labels = get_json_data(dir)

    smile_accs = {}
    token_accs = {}
    pos_accs = {}
    for k, v in data.items():
        smile_accs[k] = v['accs']['test'][0]
        token_accs[k] = v['accs']['test'][1]
        pos_accs[k] = v['accs']['test'][2]

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(12,4), sharey=True,
                                     gridspec_kw={'width_ratios': [1, 1, 2]})
    a0.bar(np.arange(len(smile_accs)), smile_accs.values(), color=colors[:len(smile_accs)])
    a0.set_xticks(np.arange(len(smile_accs)))
    a0.set_xticklabels(labels=smile_accs.keys(), rotation=45)
    a0.set_ylim([0,1])
    a0.set_ylabel('Accuracy', rotation=0, labelpad=30)
    a0.set_title('Per SMILE')
    a1.bar(np.arange(len(token_accs)), token_accs.values(), color=colors[:len(token_accs)])
    a1.set_xticks(np.arange(len(token_accs)))
    a1.set_xticklabels(labels=token_accs.keys(), rotation=45)
    a1.set_ylim([0,1])
    a1.set_title('Per Token')
    for i, set in enumerate(pos_accs.values()):
        a2.plot(set, lw=2, color=colors[i])
    a2.set_xlabel('Token Position')
    a2.set_ylim([0,1])
    a2.set_title('Per Token Sequence Position')
    return fig

def plot_moses_metrics(dir, colors=None):
    """
    Plots tiled barplot depicting the performance of the model on each MOSES metric as a function
    of epoch.

    Arguments:
        dir (str, req): Directory to json files containing calculated MOSES metrics for each model type
        colors (list): List of colors for each trained model

    """
    if colors is None:
        colors = ['#005073', '#B86953', '#932191', '#90041F', '#0F4935']

    data, labels = get_json_data(dir)
    data['paper_vae'] = {'valid': 0.977,
                         'unique@1000': 1.0,
                         'unique@10000': 0.998,
                         'FCD/Test': 0.099,
                         'SNN/Test': 0.626,
                         'Frag/Test': 0.999,
                         'Scaf/Test': 0.939,
                         'FCD/TestSF': 0.567,
                         'SNN/TestSF': 0.578,
                         'Frag/TestSF': 0.998,
                         'Scaf/TestSF': 0.059,
                         'IntDiv': 0.856,
                         'IntDiv2': 0.850,
                         'Filters': 0.997,
                         'logP': 0.121,
                         'SA': 0.219,
                         'QED': 0.017,
                         'weight': 3.63,
                         'Novelty': 0.695,
                         'runtime': 0.0}
    labels.append('paper_vae')
    metrics = list(data['paper_vae'].keys())

    fig, axs = plt.subplots(5, 4, figsize=(20,14))
    for i, ax in enumerate(fig.axes):
        metric = metrics[i]
        metric_data = []
        for label in labels:
            metric_data.append(data[label][metric])
        ax.bar(np.arange(len(metric_data)), metric_data, color=colors[:len(metric_data)])
        ax.set_xticks(np.arange(len(metric_data)))
        ax.set_xticklabels(labels=labels)
        ax.set_title(metric)
    return fig


def get_json_data(dir, fns=None, labels=None):
    """
    Opens and stores json data from a given directory

    Arguments:
        dir (str, req): Directory containing the json files
        labels (list): Labels corresponding to each file
    Returns:
        data (dict): Dictionary containing all data within
                     json files
        labels (list): List of keys corresponding to dictionary entries
    """
    if fns is None:
        fns = []
        for fn in os.listdir(dir):
            if '.json' in fn:
                fns.append(os.path.join(dir, fn))
    if labels is None:
        labels = []
        fn = fn.split('/')[-1].split('2milmoses_')[1].split('.json')[0].split('_')[0]
        labels.append(fn)

    data = {}
    for fn, label in zip(fns, labels):
        with open(fn, 'r') as f:
            dump = json.load(f)
        data[label] = dump
    return data, labels

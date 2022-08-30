import os
import sys
import abc
from typing import Optional
import torch
import model


class BaseExperiment(abc):
    def __init__(self, config: dict):
        self.experiment_config = config.get('experiment_config')
        self.gp_config = config.get('gp_config')
        self.model_config = config.get('model_config')
        self.blackbox_config = config.get('blackbox_config')
        self.dataloader_config = config.get('dataloader_config')

        modules = sys.modules[__name__]
        self.model = getattr(modules, self.model_config.get('name'))(self.model_config)
        self.gp = getattr(modules, self.gp_config.get('name'))(self.gp_config)
        self.blackbox = getattr(modules, self.blackbox_config.get('name'))(self.blackbox_config)
        self.dataloader = getattr(modules, self.dataloader_config.get('name'))(self.dataloader_config)

    def save_results(self):
        save_path = self.experiment_config['result_path']
        # TODO
        pass

    def run_experiment(self):
        # TODO
        pass

    def train_model(self):
        # TODO
        pass

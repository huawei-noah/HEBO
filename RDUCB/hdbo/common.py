#!/usr/bin/env python3
import tempfile
import os

class Config(object):
    __instance = None
    def __new__(cls, home_dir='~'):
        if Config.__instance is None:
            Config.__instance = object.__new__(cls)
            # Run only once, create the temp directory that will be used
            Config.__instance.base_path = tempfile.mkdtemp()
            Config.__instance.log_path = os.path.join(Config.__instance.base_path, 'logging')
            os.makedirs(Config.__instance.log_path)
            Config.__instance.data_path = os.path.join(Config.__instance.base_path, 'data')
            os.makedirs(Config.__instance.data_path)
            Config.__instance.ba_path = os.path.join(Config.__instance.base_path, 'bayesopt_attack')
            os.makedirs(Config.__instance.ba_path)
            Config.__instance.learnt_graphs_path = os.path.join(Config.__instance.base_path, 'learnt_graphs')
            os.makedirs(Config.__instance.learnt_graphs_path)
            # Cache to quickly compute the f_min
            Config.__instance.cache_path = os.path.expanduser(os.path.join(home_dir, 'cache'))
            os.makedirs(Config.__instance.cache_path, exist_ok=True)
            # ba_models to store the neural netowrk models
            Config.__instance.ba_models = os.path.expanduser(os.path.join(home_dir, 'ba_models'))
            os.makedirs(Config.__instance.ba_models, exist_ok=True)
            # Fcnet Function files
            Config.__instance.fcnet_path = os.path.expanduser(os.path.join(home_dir, 'fcnet'))
            os.makedirs(Config.__instance.fcnet_path, exist_ok=True)
            # MPS Function files
            Config.__instance.mps_path = os.path.expanduser(os.path.join(home_dir, 'mps'))
            os.makedirs(Config.__instance.mps_path, exist_ok=True)
        return Config.__instance
    def log_file(self, log_filename):
        return os.path.join(self.log_path, log_filename)
    def data_file(self, data_filename):
        return os.path.join(self.data_path, data_filename)
    # Saved function
    def cache_file(self, cache_filename):
        return os.path.join(self.cache_path, cache_filename)
    # Functions that are too large to store on repo and these are load only
    def fcnet_file(self, fcnet_filename):
        return os.path.join(self.fcnet_path, fcnet_filename)
    def mps_file(self, mps_filename):
        return os.path.join(self.mps_path, mps_filename)

    def list_fcnet(self):
        return os.listdir(self.fcnet_path)
    def learnt_graphs_file(self, learnt_graphs_filename):
        return os.path.join(self.learnt_graphs_path, learnt_graphs_filename)

    def full_ba_path(self):
        return Config.__instance.ba_path
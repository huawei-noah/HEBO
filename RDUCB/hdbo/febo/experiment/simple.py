import shutil

import os
from febo.experiment.experiment import ExperimentConfig, Experiment
from febo.utils import get_logger, query_yes_no
from febo.utils.config import ConfigField, assign_config, ClassConfigField
import numpy as np
import h5py

try:
    import ceramo
except ImportError:
    pass

logger = get_logger("experiment")

class SimpleExperimentConfig(ExperimentConfig):
    controller = ClassConfigField('febo.controller.SimpleController')
    environment = ClassConfigField('febo.environment.benchmarks.Camelback')
    algorithm = ClassConfigField('febo.algorithms.UCB')
    _section = 'experiment.simple'

@assign_config(SimpleExperimentConfig)
class SimpleExperiment(Experiment):
    def create(self, name=None):
        super(SimpleExperiment, self).create(name)
        self.environment = self.config.environment(path=self.directory)

    def load(self, name):
        super(SimpleExperiment, self).load(name)

        self.environment = self.config.environment(path=self.directory)
        self.algorithm = self.config.algorithm(experiment_dir=self.directory)

    def get_controller(self, remote=False, **kwargs):
        """
        returns a controller instance
        :param kwargs: arguments passed to the controller
        :return: controller instance
        """

        kwargs['experiment'] = self
        kwargs["base_path"] = kwargs.get("base_path", self._directory)
        kwargs['dbase'] =  self._dbase

        # can't use default mechanism of "get" here, as self.algorithm/self.environment attribute might not exist
        if not "algorithm" in kwargs:
            kwargs["algorithm"] = self.algorithm
        if not "environment" in kwargs:
            kwargs["environment"] = self.environment

        if remote:
            from febo.controller.remote import RemoteController
            config_cls = RemoteController
        else:
            config_cls = self.config.controller

        return config_cls(**kwargs)


    def sync(self, sync_dir):
        res = super().sync(sync_dir=sync_dir)
        with ceramo.results.Backend() as backend:

            for task_id, data in res.iterrows():
                kwargs = data['kwargs']
                run_id = kwargs['run_id']
                group_id = kwargs['group_id']

                if not self._dbase.dset_exists(group=group_id, id=run_id):
                    # The table with correct dtype should have been already created by RemoteController
                    logger.error('Sync Error: Table was not preallocated. Skipping.')
                    if query_yes_no('Do you want to set status of this entry to INVALID?', default='no'):
                        backend.store(task_id, status='INVALID')
                    continue

                # get local dataset
                dset = self._dbase.get_dset(group=group_id, id=run_id)

                if dset.size > 0:
                    logger.error('Sync Error: Local table was not empty')
                    if query_yes_no('Do you want to set status of this entry to INVALID?', default='no'):
                        backend.store(task_id, status='INVALID')
                    continue

                # get remote data
                remote_hdf5_path = os.path.join(data.directory, 'data', 'evaluations.hdf5')
                if not os.path.exists(remote_hdf5_path):
                    logger.error("Remote hdf5 file not found.")
                    continue

                remote_hdf5 = h5py.File(remote_hdf5_path, 'r')
                if not str(group_id) in remote_hdf5:
                    logger.error('Group not found in remote hdf5')
                    continue

                if not str(run_id) in remote_hdf5[str(group_id)]:
                    logger.error('table not found in remote hdf5')
                    continue

                remote_data = remote_hdf5[str(group_id)][str(run_id)][...]

                if len(remote_data) == 0:
                    logger.error('Sync Error: Remote table was empty')
                    continue

                logger.info(f"Found new remote data set (id={run_id}, group={group_id}).")

                # Copy data
                dset.add(remote_data)

                # Update status flag
                backend.store(task_id, status='SYNCED')

                # remove sync dir
                shutil.rmtree(data.directory)
                logger.info(f"Added data to local experiment and deleted files from sync directory.")

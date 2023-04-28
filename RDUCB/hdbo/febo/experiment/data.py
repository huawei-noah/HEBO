import h5py
from febo.utils.config import Configurable, Config, ConfigField, assign_config
import os
from febo.utils import  get_logger
import numpy as np

logger = get_logger("controller")

class DataBaseConfig(Config):
    chunk_size = ConfigField(200)
    _section = 'database'

@assign_config(DataBaseConfig)
class DataSet(Configurable):
    """
    Thin wrapper around a hdf5 table to store evaluations.
    """

    def __init__(self, group, id=None, dtype=None):
        """ read table `id` from group, or creates a table with name `id` if not existing. """

        if id is None:
            id = len(group)

        id = str(id)

        if not id in group:
            if dtype is None:
                raise Exception("dtype not provided and table does not exist.")

            group.create_dataset(name=id,
                                 shape=(self.config.chunk_size,),
                                 dtype=dtype,
                                 chunks=(True,),
                                 maxshape=(None,))
            self._num_data_points = 0
        else:
            self._num_data_points = len(group[id])

        self.hdf5_table = group[id]
        self._id = int(id)

    @property
    def id(self):
        return self._id


    def add(self, evaluations):
        """ add evaluation. resizes table if not large enough"""
        evaluations = np.atleast_1d(evaluations)
        n_new = evaluations.shape[0]

        if self.hdf5_table.size < self._num_data_points + n_new:
            # increase table size at least by chunk_size, or to fit all evaluations
            self.hdf5_table.resize((self.hdf5_table.size + max(self.config.chunk_size, n_new),))

        self.hdf5_table[self._num_data_points:self._num_data_points+n_new] = evaluations
        self._num_data_points += evaluations.shape[0]  # adjust counter

    @property
    def data(self):
        return self.hdf5_table[:self._num_data_points]

    @property
    def attrs(self):
        return self.hdf5_table.attrs

    @property
    def dtype(self):
        return self.hdf5_table.dtype

    @property
    def size(self):
        return self._num_data_points

    def adjust_size(self):
        """ adjust size to actual number of recorded data points"""
        self.hdf5_table.attrs["T"] = self._num_data_points
        self.hdf5_table.resize((self._num_data_points,))


class DataBase(Configurable):
    """ Thin wrapper around hdf5 file, with method to create group structure needed here. """

    def __init__(self, path):
        """ load from 'evaluations.hdf5' from path. """
        self._path = os.path.join(path, 'data')
        self._file_path = os.path.join(self._path, 'evaluations.hdf5')
        self._hdf5 = None

    def create(self):
        """ creates file, and directory if not exists """
        if not os.path.exists(self._path):
            os.mkdir(self._path)

        self._hdf5 = h5py.File(self._file_path, 'w-')
        self._hdf5.flush()

    def open(self):
        """ open hdf5 file"""
        self._hdf5 = h5py.File(self._file_path, 'r+')

    def close(self):
        """ close hdf5 file"""
        if not self._hdf5 is None:
            self._hdf5.close()

    def get_group(self, group):
        """ get existing group or create new on root level of hdf5 file. """
        self._check_dataset_loaded()
        if group is None:
            return self._hdf5

        group = str(group)

        if not group in self._hdf5:
            self._hdf5.create_group(group)

        return self._hdf5[group]

    def get_dset(self, group=None, id=None, dtype=None):
        """
        get dset in group, creates if not existing.
            returns DataSet wrapper around hdf5 table.
        """
        self._check_dataset_loaded()
        group = self.get_group(group)

        if id is None:
            id = len(group)

        return DataSet(group, id=id, dtype=dtype)

    def dset_exists(self, id, group=None):
        self._check_dataset_loaded()

        dset = self._hdf5


        if not group is None:
            group = str(group)
            if not group in dset:
                return False
            dset = dset[group]

        id = str(id)

        return id in dset

    @property
    def hdf5(self):
        return self._hdf5

    def _check_dataset_loaded(self):
        if self.hdf5 is None:
            raise Exception("Dataset not loaded. You need to call .create() or .load() first.")

    def __del__(self):
        self.close()
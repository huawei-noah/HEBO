# Created by Paul Daoudi
# Date: 11/02/2023

from setuptools import setup

setup(author='Paul Daoudi',
      name='rllg',
      version='0.1.0',
      install_requires=[
            'setuptools',
            'numpy==1.23.1',
            'torch==1.10.2',
            'tensorboardX==2.4.1',
            'mujoco-py==2.1.2.14',
            'omegaconf==2.1.1',
            'gym==0.21.0',
            'ray[tune]',
            'pyyaml',
            'matplotlib',
            'ipython',
            'pandas',
            'matplotlib',
            'jupyter',
            'ml-collections',
            'scipy'
      ]
)
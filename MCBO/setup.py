# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

from setuptools import setup

setup(
    name="MCBO",
    version="0.0.1",  # change mcbo.__init__
    author="Kamil Dreczkowski, Antoine Grosnit",
    description='# MCBO: Framework and Benchmarks for Combinatorial and Mixed-variable Bayesian Optimization',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=required,
    url='https://github.com/huawei-noah/HEBO/tree/master/MCBO',
    classifiers=[
        'License :: OSI Approved :: MIT License',
    ],
    packages=["mcbo"]
)

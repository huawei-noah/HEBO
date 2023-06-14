#!/bin/bash

# ============================================ #
#                Logic synthesis               #
# ============================================ #

EDA_DATA_ARITHMETIC_DIRPATH=./mcbo/tasks/data/epfl_benchmark/arithmetic/
mkdir -p $EDA_DATA_ARITHMETIC_DIRPATH
for CIRCUIT in 'adder' 'bar' 'div' 'hyp' 'log2' 'max' 'multiplier' 'sin' 'sqrt' 'square'; do
  wget -P "$EDA_DATA_ARITHMETIC_DIRPATH/" https://raw.githubusercontent.com/lsils/benchmarks/master/arithmetic/${CIRCUIT}.blif
done

EDA_DATA_CONTROL_DIRPATH=./mcbo/tasks/data/epfl_benchmark/random_control/
mkdir -p $EDA_DATA_CONTROL_DIRPATH
for CIRCUIT in 'arbiter' 'cavlc' 'ctrl' 'dec' 'i2c' 'int2float' 'mem_ctrl' 'priority' 'router' 'voter'; do
  wget -P "$EDA_DATA_CONTROL_DIRPATH/" https://raw.githubusercontent.com/lsils/benchmarks/master/random_control/${CIRCUIT}.blif
done

echo $EDA_DATA_ARITHMETIC_DIRPATH > ./mcbo/tasks/eda_seq_opt/circuits_path.txt
echo './libs/EDA/abc' > ./mcbo/tasks/eda_seq_opt/abc_release_path.txt

# MIG
chmod u+x ./mcbo/libs/mig_task_executable

# ============================================ #
#               RNA Inverse Fold               #
# ============================================ #
wget -P ./mcbo/tasks/data/rna_fold/ https://raw.githubusercontent.com/strevol-mpi-mis/aRNAque/v0.2/data/Eterna100/V1/eterna.csv
pip install ViennaRNA~=2.5.0a1

# ============================================ #
#               \nu-SVR - Slice                #
# ============================================ #
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00206/slice_localization_data.zip
unzip slice_localization_data.zip
rm slice_localization_data.zip
mv slice_localization_data.csv ./mcbo/tasks/data/

# ============================================ #
#               XGBoost - MNIST                #
# ============================================ #

# ============================================ #
#               Antibody design                #
# ============================================ #

# Install AbsolutNoLib
#git clone https://github.com/csi-greifflab/Absolut

# Get path associated to structures of 2DD8_S

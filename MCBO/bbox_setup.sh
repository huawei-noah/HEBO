#!/bin/bash

# ============================================ #
#                Logic synthesis               #
# ============================================ #

EDA_DATA_ARITHMETIC_DIRPATH=./mcbo/tasks/data/epfl_benchmark/arithmetic/
mkdir -p $EDA_DATA_ARITHMETIC_DIRPATH
for CIRCUIT in 'adder' 'bar' 'div' 'hyp' 'log2' 'max' 'multiplier' 'sin' 'sqrt' 'square'; do
  wget -P "$EDA_DATA_ARITHMETIC_DIRPATH/" https://raw.githubusercontent.com/lsils/benchmarks/master/arithmetic/${CIRCUIT}.blif
  wget -P "$EDA_DATA_ARITHMETIC_DIRPATH/" https://raw.githubusercontent.com/lsils/benchmarks/master/arithmetic/${CIRCUIT}.aig
done

EDA_DATA_CONTROL_DIRPATH=./mcbo/tasks/data/epfl_benchmark/random_control/
mkdir -p $EDA_DATA_CONTROL_DIRPATH
for CIRCUIT in 'arbiter' 'cavlc' 'ctrl' 'dec' 'i2c' 'int2float' 'mem_ctrl' 'priority' 'router' 'voter'; do
  wget -P "$EDA_DATA_CONTROL_DIRPATH/" https://raw.githubusercontent.com/lsils/benchmarks/master/random_control/${CIRCUIT}.blif
  wget -P "$EDA_DATA_CONTROL_DIRPATH/" https://raw.githubusercontent.com/lsils/benchmarks/master/random_control/${CIRCUIT}.aig
done

ABC_EXE='./libs/EDA/abc'
echo $EDA_DATA_ARITHMETIC_DIRPATH >./mcbo/tasks/eda_seq_opt/circuits_path.txt
echo $ABC_EXE >./mcbo/tasks/eda_seq_opt/abc_release_path.txt
chmod u+x $ABC_EXE

# MIG
chmod u+x ./libs/EDA/mig_task_executable

# ============================================ #
#               RNA Inverse Fold               #
# ============================================ #
wget -P ./mcbo/tasks/data/rna_fold/ https://raw.githubusercontent.com/strevol-mpi-mis/aRNAque/v0.2/data/Eterna100/V1/eterna.csv
pip install ViennaRNA~=2.5.0a1
# If `import RNA` fails: try to reinstall ViennaRNA from scratch (remove the wheel)

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
# ==> NOTHING

# ============================================ #
#               Antibody design                #
# ============================================ #

# Install AbsolutNoLib
wget https://github.com/csi-greifflab/Absolut/archive/a50b3e41e2b7170aee207f067cf8d7009234c30e.zip -P ./libs/
cd ./libs/
unzip a50b3e41e2b7170aee207f067cf8d7009234c30e.zip
rm a50b3e41e2b7170aee207f067cf8d7009234c30e.zip
mv Absolut-a50b3e41e2b7170aee207f067cf8d7009234c30e Absolut
cd Absolut/src
make

# Get path associated to structures of 2DD8_S
mkdir -p ./antigen_data/2DD8_S
cd ./antigen_data/2DD8_S
wget https://ns9999k.webs.sigma2.no/10.11582_2021.00063/projects/NS9603K/pprobert/AbsolutOnline/Structures/SUDDL6142d1af0c3837a24ca534e96b7192bb-10-11-af8a57bbcc249709ce1058ec80ba20a4Structures.txt.zip
unzip SUDDL6142d1af0c3837a24ca534e96b7192bb-10-11-af8a57bbcc249709ce1058ec80ba20a4Structures.txt.zip
rm SUDDL6142d1af0c3837a24ca534e96b7192bb-10-11-af8a57bbcc249709ce1058ec80ba20a4Structures.txt.zip

# go back to project root and add path to executable
cd ../../../../../
echo './libs/Absolut/src/' > /data/antoineg/MCBO/mcbo/tasks/antibody_design/path_to_AbsolutNoLib.txt

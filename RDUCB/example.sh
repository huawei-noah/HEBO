# Run two seeds of random search on example1 problem (Rosenbrock)
mlflow run . -P param_file=config/examples/example1_random.yml -P seed=0
mlflow run . -P param_file=config/examples/example1_random.yml -P seed=1

# Run two seeds of RDUCB on example1 problem (Rosenbrock)
mlflow run . -P param_file=config/examples/example1_rducb.yml -P seed=0
mlflow run . -P param_file=config/examples/example1_rducb.yml -P seed=1

# Plot the runs specified in example_experiment1.json of length 10 and include a legened
python plot.py example_experiment1.json 10 --legend

# Run two seeds of REMBO on example2 problem (Lasso) and select Diabetes sub-problem
mlflow run . -P param_file=config/examples/example2_rembo.yml -P seed=0 -P sub_benchmark="pick_data:diabetes"
mlflow run . -P param_file=config/examples/example2_rembo.yml -P seed=1 -P sub_benchmark="pick_data:diabetes"

# Run two seeds of RDUCB on example2 problem (Lasso) and select Diabetes sub-problem
mlflow run . -P param_file=config/examples/example2_rducb.yml -P seed=0 -P sub_benchmark="pick_data:diabetes"
mlflow run . -P param_file=config/examples/example2_rducb.yml -P seed=1 -P sub_benchmark="pick_data:diabetes"

# Plot the runs specified in example_experiment2.json of length 25, include a legened and add scientific notation on y-axis
python plot.py example_experiment2.json 25 --legend --sci
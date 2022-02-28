# Combinatorial Optimization with Simulated Annealing
The algorithm takes an initial input design and try to optimize it using a simulated annealing approach, where the temperature changes over the iteration and the target metric (area/delay) is modified accordingly.

## How to run
- Install dependencies: `pip3 install pyyaml joblib`
- Edit `data.yml` file to specify your design file, library file, output directory and modify other parameters
- Run using: `python3 simulated-annealing.py data.yml`
- Logs and results are written to the `output_dir` specified in the `data.yml` file.

# Greedy Combinatorial Optimization
The algorithm takes an initial input design and spawns parallel threads to perform each of the given transformations on the design. Then, it keeps the design with the minimum area for the next iteration, whether it meets the delay constraint or not. After that, it continues until no improvements in the design area is made.

## How to run
- Install dependencies: `pip3 install pyyaml joblib`
- Edit `data.yml` file to specify your design file, library file, output directory and modify other parameters
- Run using: `python3 greedy.py data.yml`
- Logs and results are written to the `output_dir` specified in the `data.yml` file.


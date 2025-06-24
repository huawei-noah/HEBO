import time

from task.tools import Absolut
from utilities.config_utils import load_config

from tqdm import tqdm

from random_search.optimizer import Optimizer
from utilities.results_logger import ResultsLogger

if __name__ == '__main__':
    config = load_config('./random_search/config.yaml')
    absolut_config = {"antigen": "S3A3_A",
                      "path": config['absolut_config']['path'],
                      "process": config['absolut_config']['process'],
                      'startTask': config['absolut_config']['startTask']}

    # Defining the fitness function
    absolut_binding_energy = Absolut(absolut_config)


    def black_box(x):
        x = x.astype(int)
        return absolut_binding_energy.energy(x)


    optim = Optimizer(config)
    results = ResultsLogger(config['rs_num_iter'])
    config['rs_batch_size'] = 100

    bb_evals = 0  # TODO this will be kept track off in the Absolut class or Env class

    for itern in tqdm(range(int(config['rs_num_iter'] / config['rs_batch_size']))):
        start = time.time()
        x_next = optim.suggest(config['rs_batch_size'])  # todo note that the shape of this is (batch_size, seq_len)
        end = time.time()

        y_next = black_box(x_next)  # todo same here
        bb_evals += config['rs_batch_size']

        optim.observe(x_next, y_next)

        results.append(x_next, y_next, end - start, bb_evals)

    results.save(config['save_dir'])
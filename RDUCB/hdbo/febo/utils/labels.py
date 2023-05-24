
def algorithm_optimizer_gradients(id, config):
    algorithm = config['experiment.simple']['algorithm'].rsplit('.', 1)[1]
    use_gradients = '-grad' if config['optimizer.scipy']['lbfgs_use_gradients'] else ''
    processes = config['optimizer.scipy']['num_processes']
    sync = '-sync' if  config['optimizer.scipy']['sync_restarts'] else ''
    optimizer = config['algorithm']['optimizer'].rsplit('.', 1)[1]

    return f"{algorithm}-{optimizer}{use_gradients}-{processes}{sync}"

def environment(id, config):
    env = config['experiment.simple']['environment'].rsplit('.', 1)[1]
    return f"{id}-{env}"

def environment(id, config):
    env = config['experiment.simple']['environment'].rsplit('.', 1)[1]
    return f"{id}-{env}"

def noise_test(id, config):
    model = config['algorithm']['model'].rsplit('.', 1)[1]
    algorithm = config['experiment.simple']['algorithm'].rsplit('.', 1)[1]

    return f"{id}-{algorithm}-{model}"


def algorithm(id, config):
    algorithm = config['experiment.simple']['algorithm'].rsplit('.', 1)[1]
    env = config['experiment.simple']['environment'].rsplit('.', 1)[1]

    return f"{id}-{algorithm}-{env}"

def algorithm_name(id, config):
    algorithm = config['experiment.simple']['algorithm'].rsplit('.', 1)[1]
    # env = config['experiment.simple']['environment'].rsplit('.', 1)[1]

    return f"{id}-{algorithm}"
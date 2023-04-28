def algorithm_name(id, config):
    algorithm = config['experiment.simple']['algorithm'].rsplit('.', 1)[1]
    # env = config['experiment.simple']['environment'].rsplit('.', 1)[1]
    tr_radius = get_setting(config, 'algorithm.subdomainbo', 'tr_radius')
    beta = get_setting(config, 'model', 'beta')
    tr_method = get_setting(config, 'algorithm.subdomainbo','tr_method')
    max_queries_tr = get_setting(config, 'algorithm.subdomainbo', 'max_queries_tr')


    acquisition = ''
    if 'algorithm.subdomainbo' in config and 'acquisition' in config['algorithm.subdomainbo']:
        acquisition = f"-{config['algorithm.subdomainbo']['acquisition'].rsplit('.', maxsplit=1)[1]}"

    return f"{id}-{algorithm}{tr_radius}{tr_method}{max_queries_tr}{acquisition}{beta}"

def get_setting(config, section, setting):
    if section in config and setting in config[section]:
        return f"-{config[section][setting]}"
    return ''
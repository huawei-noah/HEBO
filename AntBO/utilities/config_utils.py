import yaml


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config, save_path):
    with open(save_path, 'w') as file:
        documents = yaml.dump(config, file)
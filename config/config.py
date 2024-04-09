import yaml
from box import Box

def get_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return Box(config)
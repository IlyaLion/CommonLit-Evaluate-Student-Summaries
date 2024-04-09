import warnings
import os

from training.training import train_model
from config.config import get_config

def main():
    warnings.filterwarnings("ignore")

    config = get_config(os.path.join('config', 'config.yaml'))
    val_scores = train_model(config=config, checkpoint_dir=config.directories.models)
    print(f'{val_scores=}')

if __name__ == '__main__':
    main()
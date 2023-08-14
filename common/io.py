from typing import Dict
import os

import yaml


def load_config_from_yaml(cfg_file : str, verbose : bool=True) -> Dict:
    """Load YAML config"""
    if not os.path.exists(cfg_file):
        if verbose:
            print(f'{cfg_file} does not exist.')
        return None
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
    if verbose:
        print(f'config at {cfg_file} loaded')

    return cfg

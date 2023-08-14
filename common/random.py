import random
import numpy as np
import torch


def set_random_seeds(seed : int=73, rank : int=0):
    """
    Set random seed according to seed and (process) rank
    """
    seed_value = seed + rank  # TODO: consider moving to a slightly better function
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)   
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

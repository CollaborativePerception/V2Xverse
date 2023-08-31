from typing import Union
import os

import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


def load_checkpoint(
    checkpoint_path : str,
    device : Union[str, torch.device],
    model : nn.Module,
    optimizer : Union[torch.optim.Optimizer, None]=None,
    lr_scheduler : Union[LRScheduler, None]=None,
) -> int:
    """
    Load checkpoint from parsed args
    TODO (yinda): define data schema for training context to avoid returning specific type (like a single int as for now)
    Reference: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training

    Args:
        checkpoint_path: path to checkpoint for resuming
        device: device which all elements (model, optimizer, etc.) will be mapped to
        model: torch model whose state to be restored from checkpoint
            NOTE: need to be a CPU-based standalone model (e.g. not wrapped by DDP)
        optimizer: optimizer whose state to be restored from checkpoint
        lr_scheduler: learning rate scheduler whose state to be restored from checkpoint

    Return:
        epoch index
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint['model_state_dict'], 'module.')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    epoch_idx = checkpoint['epoch']

    LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
    print(f'Rank {LOCAL_RANK} loaded checkpoint from: {checkpoint_path}')

    return epoch_idx

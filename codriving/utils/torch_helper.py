from typing import Dict, Union
import torch
from torch.utils.data import DataLoader

from codriving import CODRIVING_REGISTRY
from common.registry import build_object_within_registry_from_config


def move_dict_data_to_device(
    data : Dict[str, torch.Tensor],
    device : Union[str, torch.device],
    ):
    """
    Move data to a specific device. Assume data contained in a dicionary

    Args:
        data: data contained in a flattend (non-hierarchical) dicionary.
        device: destination device of the data
    """
    for k in data:
        data[k] = data[k].to(device)


def build_dataloader(
    config : Dict,
    is_distributed : bool,
    ) -> DataLoader:
    """
    Build dataloader given config

    Args:
        config: dict config in the following format

            .. code-block::

                config
                | -- dataset: ...
                | -- distributed_sampler: ...
                | -- dataloader: ...
        is_distributed: whether running in distributed environment

    Return:
        DataLoader: built dataloader
    """
    # dataset
    dataset_config = config['dataset']
    dataset = build_object_within_registry_from_config(
        CODRIVING_REGISTRY,
        dataset_config,
    )
    # sampler
    if is_distributed:
        distributed_sampler_config = config['distributed_sampler']
        data_sampler = build_object_within_registry_from_config(
            CODRIVING_REGISTRY,
            distributed_sampler_config,
            dataset=dataset,
        )
    else:
        data_sampler = None
    # dataloader
    dataloader_config = config['dataloader']
    dataloader = build_object_within_registry_from_config(
        CODRIVING_REGISTRY,
        dataloader_config,
        dataset=dataset,
        sampler=data_sampler,
        collate_fn=dataset.collate_fn,
    )

    return dataloader

from typing import Dict, Union
import torch


def move_dict_data_to_device(
    data : Dict[str, torch.Tensor],
    device : Union[str, torch.device],
    ):
    """
    Move data to a specific device
    Assume data contained in a dicionary

    Args:
        data (Dict): data contained in a flattend (non-hierarchical) dicionary.
        device (Union[str, torch.device]): destination device of the data
    """
    for k in data:
        data[k] = data[k].to(device)

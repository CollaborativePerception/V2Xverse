from typing import Union

import torch
from torch import nn

def decorate_model(
    model: nn.Module,
    clip_grad: Union[None, float]= None,
    ):
    if clip_grad is not None:
        assert clip_grad > 0, f"positive clip_grad expected, {clip_grad} get"
        # Reference: https://stackoverflow.com/a/54816498
        for p in model.parameters():
            p.register_hook(
                lambda grad: torch.clamp(grad, -clip_grad, clip_grad))

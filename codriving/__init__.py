import inspect

import torch
import torch.utils.data
from torch import optim
from torch import nn

from common import Registry
CODRIVING_REGISTRY = Registry('codriving')  # TODO (yinda): resolve this order-sensitive code
import codriving.data_utils
import codriving.models
import codriving.losses


def _register_all_classes_within_module(m):
    for k, v in m.__dict__.items():
        # filter out private objects and magic methods
        if k.startswith('_'):
            continue
        # filter out non-class object,
        #   assuming python naming convention being strictly followed within PyTorch
        if not k[0].isupper():
            continue
        if k in CODRIVING_REGISTRY:
            continue
        if not inspect.isclass(v):
            continue
        if not inspect.isclass(v):
            continue
        if v.__name__ in CODRIVING_REGISTRY:
            continue
        CODRIVING_REGISTRY.register(v)


# register all optimizers from torch
_register_all_classes_within_module(optim)
# register all lr_schedulers from torch
_register_all_classes_within_module(optim.lr_scheduler)
# register all nn.Modules from torch
_register_all_classes_within_module(nn)
# register all torch.utils.data from torch
_register_all_classes_within_module(torch.utils.data)

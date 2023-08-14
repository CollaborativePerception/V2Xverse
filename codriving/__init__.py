from torch import optim
from codriving.utils import Registry


CODRIVING_REGISTRY = Registry('codriving')

# register all optimizers from torch
for k, v in optim.__dict__.items():
    # filter out private objects and magic methods
    if k.startswith('_'):
        continue
    CODRIVING_REGISTRY.register(v)

# register all nn.Modules from torch
for k, v in optim.__dict__.items():
    # filter out private objects and magic methods
    if k.startswith('_'):
        continue
    CODRIVING_REGISTRY.register(v)

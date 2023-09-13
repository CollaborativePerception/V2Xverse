from typing import Tuple, Dict, Any

import torch
from torch import nn

from common.registry import build_object_within_registry_from_config
from codriving import CODRIVING_REGISTRY


@CODRIVING_REGISTRY.register
class CompoundLoss(nn.Module):
    """
    Compound loss, useful for weighting multiple losses
    """
    def __init__(self, loss_configs : Dict):
        """
        Args:
            loss_configs: config of losses
        """
        super(CompoundLoss, self).__init__()
        self.loss_names = list()
        self.loss_modules = nn.ModuleList()
        self.loss_weights = list()
        for loss_item in loss_configs:
            self.loss_names.append(loss_item['config']['type'])
            self.loss_modules.append(build_object_within_registry_from_config(
                CODRIVING_REGISTRY, loss_item['config']))
            self.loss_weights.append(loss_item['weight'])

    def forward(
            self,
            batch_data : Dict,
            model_output : Dict,
            ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward behavior

        Args:
            batch_data: loaded batch data
            model_output: output from model

        Return:
            Tuple[torch.Tensor, Dict]:
            - first element: loss to be back propagated
            - second element: extra information
        """
        reduced_loss = 0
        detached_losses = dict()
        for loss_name, loss_module, loss_weight in \
            zip(self.loss_names, self.loss_modules, self.loss_weights):
            loss, _ = loss_module(batch_data, model_output)
            reduced_loss += loss
            detached_losses[loss_name] = loss.detach().cpu().numpy()

        return loss, detached_losses

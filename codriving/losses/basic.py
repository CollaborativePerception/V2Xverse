from typing import Dict

import torch
from torch import nn

from common.registry import build_object_within_registry_from_config


class CompoundLoss(nn.Module):
    def __init__(self, loss_configs):
        super(CompoundLoss, self).__init__()
        self.loss_names = list()
        self.loss_modules = nn.ModuleList()
        self.loss_weights = list()
        for loss_item in loss_configs:
            self.loss_names(loss_item['name'])
            self.loss_modules.append(
                build_object_within_registry_from_config(loss_item['config'])
                )
            self.loss_weights.append(loss_item['weight'])

    def forward(self, batch_data : Dict, model_output : Dict):
        """
        Forward behavior

        Args:
            batch_data (Dict): loaded batch data
            model_output (Dict): output from model

        Returns:
            torch.Tensor: loss to be back propagated
            Dict: extra information
        """
        reduced_loss = 0
        detached_losses = dict()
        for loss_name, loss_module, loss_weight in \
            zip(self.loss_names, self.loss_modules, self.loss_weights):
            loss, _ = loss_module(batch_data, model_output)
            reduced_loss += loss
            detached_losses[loss_name] = loss.detach().cpu().numpy()

        return loss, detached_losses

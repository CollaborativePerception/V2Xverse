from typing import Tuple, Dict, Any

import torch
from torch import nn

from codriving import CODRIVING_REGISTRY


@CODRIVING_REGISTRY.register
class WaypointL1Loss(nn.Module):
    """Loss for supervising waypoint predictor
    """
    def __init__(self, l1_loss=torch.nn.L1Loss):
        super(WaypointL1Loss, self).__init__()
        self.loss = l1_loss(reduction="none")
        # TODO: remove this hardcode
        # and make it extensible to variable trajectory length
        self.weights = [
            0.1407441030399059,
            0.13352157985305926,
            0.12588535273178575,
            0.11775496498388233,
            0.10901991343009122,
            0.09952110967153563,
            0.08901438656870617,
            0.07708872007078788,
            0.06294267636589287,
            0.04450719328435308,
        ]

    def forward(self,
                batch_data : Dict,
                model_output : Dict,
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute loss

        Args:
            batch_data: loaded batch data
            model_output: output from model

        Return:
            Tuple[torch.Tensor, Dict]:
            - first element: loss to be back propagated
            - second element: extra information
        """
        output = model_output['future_waypoints']
        target = batch_data['future_waypoints']
        loss = self.loss(output, target)  # shape: n, 12, 2
        loss = torch.mean(loss, (0, 2))  # shape: 12
        loss = loss * torch.tensor(self.weights, device=output.device)
        extra_info = dict()

        return torch.mean(loss), extra_info

@CODRIVING_REGISTRY.register
class ADE_FDE(nn.Module):
    """Loss for supervising waypoint predictor
    """
    def __init__(self, l1_loss=torch.nn.L1Loss):
        super(ADE_FDE, self).__init__()

    def forward(self,
                batch_data : Dict,
                model_output : Dict,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss

        Args:
            batch_data: loaded batch data
            model_output: output from model

        Return:
            Tuple[torch.Tensor, torch.Tensor]:
            - first element: ADE
            - second element: FDE
        """
        output = model_output['future_waypoints']
        target = batch_data['future_waypoints']

        dis = (torch.sum((output - target)**2, (2)))**0.5 # shape: n, 10
        ADE = torch.mean(dis, (1)) # shape: n
        FDE = torch.mean(dis[:,-1:], (1)) # shape: n

        return ADE, FDE
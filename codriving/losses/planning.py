import torch

from codriving import CODRIVING_REGISTRY


@CODRIVING_REGISTRY.register
class WaypointL1Loss:
    """
    Loss for supervising waypoint predictor
    """
    def __init__(self, l1_loss=torch.nn.L1Loss):
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

    def __call__(self, output, target):
        # invaild_mask = target.ge(1000)
        # output[invaild_mask] = 0
        # target[invaild_mask] = 0
        loss = self.loss(output, target)  # shape: n, 12, 2
        loss = torch.mean(loss, (0, 2))  # shape: 12
        loss = loss * torch.tensor(self.weights, device=output.device)
        return torch.mean(loss)

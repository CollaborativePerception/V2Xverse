from typing import Iterable
from typing import Tuple, Dict

from codriving import CODRIVING_REGISTRY

import torch.nn.functional as F
import torch.nn as nn
import torch
import logging
_logger = logging.getLogger(__name__)


class Conv3D(nn.Module):
    def __init__(self, in_channel : int, out_channel : int, kernel_size : int, stride : int, padding : int):
        super(Conv3D, self).__init__()
        self.conv3d = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3d = nn.BatchNorm3d(out_channel)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # input x: (batch, seq, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, c, seq_len, h, w)
        x = F.relu(self.bn3d(self.conv3d(x)))
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, seq_len, c, h, w)

        return x
    

class MLP(nn.Module):
	def __init__(self, in_feat : int, out_feat : int, hid_feat : Iterable[int]=(1024, 512), activation=None, dropout=-1):
		super(MLP, self).__init__()
		dims = (in_feat, ) + hid_feat + (out_feat, )

		self.layers = nn.ModuleList()
		for i in range(len(dims) - 1):
			self.layers.append(nn.Linear(dims[i], dims[i + 1]))

		self.activation = activation if activation is not None else lambda x: x
		self.dropout = nn.Dropout(dropout) if dropout != -1 else lambda x: x

	def forward(self, x : torch.Tensor) -> torch.Tensor:
		for i in range(len(self.layers)):
			x = self.activation(x)
			x = self.dropout(x)
			x = self.layers[i](x)
		return x


@CODRIVING_REGISTRY.register
class WaypointPlanner(nn.Module):
    """
    TODO (yinda): remove and make parameters configurable
    """
    def __init__(self,):
        super().__init__()
        height_feat_size = 6
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)

        self.decoder = MLP(256+128, 20, hid_feat=(1025, 512))

        self.target_encoder = MLP(2, 128, hid_feat=(16, 64))


    def reset_parameters(self):
        pass
    

    def forward(self, input_data : Dict) -> torch.Tensor:
        """Forward method for WaypointPlanner

        Args:
            input_data: input data to forward

                required keys:

                - occupancy: rasterized map from perception results
                - target: target point to go to

        Return:
            torch.Tensor: predicted waypoints
        """
        occupancy = input_data["occupancy"]  # B,T,C,H,W = [B, 5, 6, 384, 192]
        # print("occupancy_map shape: ", occupancy_map.shape)
        # [4, 5, 2, 40, 20]
        batch, seq, c, h, w = occupancy.size()

        x = occupancy.view(-1, c, h, w)  # batch*seq, c, h, w
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))
        # print("X_0 1: ", x.shape)
        # [20, 32, 40, 20]
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))
        # print("X_0 2: ", x.shape)
        # [20, 32, 40, 20]


        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        # print("X_1 1: ", x_1.shape)
        # [20, 64, 20, 10]
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        # print("X_1 1: ", x_1.shape)
        # [20, 64, 20, 10]

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_1 = self.conv3d_1(x_1)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w)

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_2 = self.conv3d_2(x_2)
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1
        # print(x_2.shape)
        # [4, 128, 10, 5]

        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))
        # print(x_3.shape)
        # [4, 256, 10, 5]

        feature = x_3.mean(dim=(2, 3))  # NOTE: adopt the mean pooling!
        # 4, 256
        # print(feature.shape, input_data['target'].shape)
        feature_target = self.target_encoder(input_data['target'])
        # print(feature_target.shape)
        future_waypoints = self.decoder(torch.cat((feature, feature_target), dim=1)).contiguous().view(batch, 10, 2)
        output_data = dict(future_waypoints=future_waypoints)

        return output_data


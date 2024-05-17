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
class WaypointPlanner_e2e(nn.Module):
    """
    WaypointPlanner with BEV feature, occupancy map and road map as inputs
    """
    def __init__(self,feature_dir=128):
        super().__init__()
        height_feat_size = 6
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        ####### for extral feature
        self.conv_pre_1_f = nn.Conv2d(feature_dir, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2_f = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1_f = nn.BatchNorm2d(32)
        self.bn_pre_2_f = nn.BatchNorm2d(32)

        self.conv_pre_1_f2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2_f2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1_f2 = nn.BatchNorm2d(32)
        self.bn_pre_2_f2 = nn.BatchNorm2d(32)
        #######

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
        occupancy = input_data["occupancy"]  # B,T,C,H,W = ([2, 5, 6, 192, 96])
        batch, seq, c, h, w = occupancy.size()

        x = occupancy.view(-1, c, h, w)  # batch*seq, c, h, w ([10, 6, 192, 96])
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x))) # ([batch*seq, 32, 192, 96])

        ########### feature embedding ##############
        features_list = input_data['feature_warpped_list']
        feature = torch.cat(features_list, dim=0)

        batch, seq, c2, h, w = feature.size()

        xf = feature.view(-1, c2, h, w)  # batch*seq, c, h, w ([10, 128, 192, 96])
        xf = F.relu(self.bn_pre_1_f(self.conv_pre_1_f(xf))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        xf = F.relu(self.bn_pre_2_f(self.conv_pre_2_f(xf))) # ([batch*seq, 32, 192, 96])

        # concatenate feature and embedd again
        x_enhanced = torch.cat((x,xf), dim=1)  # batch*seq, c, h, w ([10, 64, 192, 96])
        x_enhanced = F.relu(self.bn_pre_1_f2(self.conv_pre_1_f2(x_enhanced))) # ([batch*seq, 32, 192, 96]) ([10, 32, 192, 96])
        x_enhanced = F.relu(self.bn_pre_2_f2(self.conv_pre_2_f2(x_enhanced))) # ([batch*seq, 32, 192, 96])



        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        # x_1 = F.relu(self.bn1_1(self.conv1_1(x))) # ([10, 64, 96, 48])
        x_1 = F.relu(self.bn1_1(self.conv1_1(x_enhanced)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1))) # ([10, 64, 96, 48])


        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)  ([2, 5, 64, 96, 48])
        x_1 = self.conv3d_1(x_1)  # ([2, 3, 64, 96, 48])
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w) ([6, 64, 96, 48])

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1))) # ([6, 128, 48, 24])
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2))) # ([6, 128, 48, 24])

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w) ([2, 3, 128, 48, 24])
        x_2 = self.conv3d_2(x_2)  #  ([2, 1, 128, 48, 24])
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1 ([2, 128, 48, 24])

        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))  #  ([2, 256, 24, 12])

        feature = x_3.mean(dim=(2, 3))  # NOTE: adopt the mean pooling!  ([2, 256])
        feature_target = self.target_encoder(input_data['target'])  #  ([2, 128])
        future_waypoints = self.decoder(torch.cat((feature, feature_target), dim=1)).contiguous().view(batch, 10, 2)  #  ([2, 10, 2])
        output_data = dict(future_waypoints=future_waypoints)

        return output_data


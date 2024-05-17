
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import DeformConv2dPack as dconv2d
from timm.models.layers import DropPath
from opencood.models.sub_modules.cbam import BasicBlock
from opencood.models.sub_modules.feature_alignnet_modules import SCAligner, Res1x1Aligner, \
    Res3x3Aligner, Res3x3Aligner, CBAM, ConvNeXt, FANet, SDTAAgliner
import numpy as np



class AlignNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_name = args['core_method']
        
        if model_name == "scaligner":
            self.channel_align = SCAligner(args['args'])
        elif model_name == "resnet1x1":
            self.channel_align = Res1x1Aligner(args['args'])
        elif model_name == "resnet3x3":
            self.channel_align = Res3x3Aligner(args['args'])
        elif model_name == "sdta":
            self.channel_align = SDTAAgliner(args['args'])
        elif model_name == "cbam":
            self.channel_align = CBAM(args['args'])
        elif model_name == "convnext":
            self.channel_align = ConvNeXt(args['args'])
        elif model_name == "fanet":
            self.channel_align = FANet(args['args'])
        elif model_name == 'identity':
            self.channel_align = nn.Identity()

        self.spatial_align_flag = args.get("spatial_align", False)
        if self.spatial_align_flag:
            warpnet_indim = args['args']['warpnet_indim']
            dim = args['args']['dim']
            self.teacher = args['args']['teacher']
            setattr(self, "warpnet", 
                nn.Sequential(
                nn.Conv2d(warpnet_indim, warpnet_indim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(warpnet_indim),
                nn.ReLU(),
                nn.Conv2d(warpnet_indim, dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                nn.Conv2d(dim, 2, kernel_size=3, stride=1, padding=1),
                )
            )
            self.theta_identity = torch.tensor([[[1.,0.,0.],[0.,1.,0.]]])

        self.count = 0 # debug

    def forward(self, x):
        return self.channel_align(x)


    def spatail_align(self, student_feature, teacher_feature, physical_dist):
        physical_offset = self.warpnet(torch.cat([student_feature, teacher_feature], dim=1)).permute(0,2,3,1) # N, H, W, 2, unit is meter.
        mask = torch.any(teacher_feature != 0, dim=1)
        physical_offset *= mask.unsqueeze(-1)
        relative_offset = physical_offset * torch.tensor([2./physical_dist[0], 2./physical_dist[1]], device=physical_offset.device)  # N, H, W, 2
        warp_field = relative_offset + \
            torch.nn.functional.affine_grid(self.theta_identity.expand(student_feature.shape[0], 2, 3), student_feature.shape).to(relative_offset.device)
        spataial_aligned_feature = torch.nn.functional.grid_sample(student_feature, warp_field)

        # self.visualize_offset(physical_offset, warp_field, student_feature, spataial_aligned_feature, teacher_feature)
        return spataial_aligned_feature

    def visualize_offset(self, physical_offset, warp_field, feature_before, feature_after, teacher_feature):
        """
        physical_offset: shape [N, H, W, 2]
        warp_field: shape [N, H, W, 2]
        feaure_before: [N, C, H, W]
        feature_after: [N, C, H, W]
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        import os
        N = physical_offset.shape[0]
        print(physical_offset.shape)
        
        save_path = "opencood/logs/vislog"
        file_idx = self.count
        self.count += 1

        physical_offsets_save_path = os.path.join(save_path, "physical_offsets")
        vmax = physical_offset.max()
        print(f"physical offset max: {vmax}")
        if not os.path.exists(physical_offsets_save_path):
            os.mkdir(physical_offsets_save_path)
        physical_offset = physical_offset.detach().cpu().numpy()
        warp_field = warp_field.detach().cpu().numpy()
        for i in range(N):
            sns.heatmap(physical_offset[i,:,:,0], cmap="vlag", vmin=-vmax*0.8, vmax=vmax*0.8, square=True)
            plt.axis('off')
            plt.savefig(os.path.join(physical_offsets_save_path, "{}_{}_physical_x.png".format(file_idx, i)), dpi=500)
            plt.close()

            sns.heatmap(physical_offset[i,:,:,1], cmap="vlag", vmin=-vmax*0.8, vmax=vmax*0.8, square=True)
            plt.axis('off')
            plt.savefig(os.path.join(physical_offsets_save_path, "{}_{}_physical_y.png".format(file_idx, i)), dpi=500)
            plt.close()

            sns.heatmap(warp_field[i,:,:,0], cmap="vlag", square=True)
            plt.axis('off')
            plt.savefig(os.path.join(physical_offsets_save_path, "{}_{}_warpfield_x.png".format(file_idx, i)), dpi=500)
            plt.close()

            sns.heatmap(warp_field[i,:,:,1], cmap="vlag", square=True)
            plt.axis('off')
            plt.savefig(os.path.join(physical_offsets_save_path, "{}_{}_warpfield_y.png".format(file_idx, i)), dpi=500)
            plt.close()

        spatial_feature_save_path = os.path.join(save_path, "spatial_feature")
        if not os.path.exists(spatial_feature_save_path):
            os.mkdir(spatial_feature_save_path)
        feature_before = feature_before.detach().cpu().numpy()
        feature_after = feature_after.detach().cpu().numpy()
        teacher_feature = teacher_feature.detach().cpu().numpy()
        for i in range(N):
            channel = np.random.randint(64)
            plt.imshow(feature_before[i, channel])
            plt.axis("off")
            plt.colorbar()
            plt.savefig(os.path.join(spatial_feature_save_path, "{}_{}_before.png".format(file_idx, i)), dpi=500)
            plt.close()

            plt.imshow(feature_after[i, channel])
            plt.axis("off")
            plt.colorbar()
            plt.savefig(os.path.join(spatial_feature_save_path, "{}_{}_spaligned.png".format(file_idx, i)), dpi=500)
            plt.close()
            
            plt.imshow(teacher_feature[i, channel])
            plt.axis("off")
            plt.colorbar()
            plt.savefig(os.path.join(spatial_feature_save_path, "{}_{}_teacher.png".format(file_idx, i)), dpi=500)
            plt.close()
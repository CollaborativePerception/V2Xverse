import torch
import torch.nn as nn
from opencood.models.da_modules.gsl import GradientScalarLayer

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.indim = args['indim']
        self.roi_size = args['roi_align_size']
        self.netD = nn.Sequential(
            nn.Conv2d(self.indim, self.indim//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.indim//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.indim//2, self.indim//4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.indim//4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(kernel_size=self.roi_size, stride=1, padding=0), # [N, self.indim//4, 1, 1],
            nn.Flatten(start_dim=1),
            nn.Linear(self.indim//4, self.indim//8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.indim//8, 1),
            nn.Sigmoid()
        )
        self.grl = GradientScalarLayer(- args.get('scale', 1))

        self.netD.apply(weights_init)
    
    def forward(self, x):
        """
        Input:
            x: [N, indim, RoIsize, RoIsize]
        Output:
            cls: [N, 1]
        """
        x = self.grl(x)
        return self.netD(x)
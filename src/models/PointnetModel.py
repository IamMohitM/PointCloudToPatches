import torch as th
import torch.nn as nn
from src.models.pointnet_utils import PointNetEncoder

class ReconstructionModel(nn.Module):
    """Point Cloud to patch model that encodes an Point Clouds and outputs parameters."""

    def __init__(self, output_dim, init=None):
        """Construct the model according to the necessary output dimension.

        params:
            output_dim - output patch dimension?
        """
        super(ReconstructionModel, self).__init__()
        self.encode = PointNetEncoder(global_feat=True, feature_transform=True, channel=6)
        self.decode = nn.Sequential(
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
        )
        self.out = nn.Linear(256, output_dim)
        if init is not None:
            nn.init.zeros_(self.out.weight)
            with th.no_grad():
                self.out.bias.data = init.clone()

    def forward(self, pc):
        """Process one our more images, corresponding to different views."""
        x, trans, trans_feat = self.encode(pc)
        decoder = self.decode(x)
        params = self.out(decoder)
        return params

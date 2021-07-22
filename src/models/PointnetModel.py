import torch as th
import torch.nn as nn


class ReconstructionModel(nn.Module):
    """Point Cloud to patch model that encodes an Point Clouds and outputs parameters."""

    def __init__(self, args, output_dim, init=None):
        """Construct the model according to the necessary output dimension.

        params:
            output_dim - output patch dimension?
        """
        self.encoder = args.encoder.lower()
        super(ReconstructionModel, self).__init__()
        if self.encoder == "pointnet":
            from src.models.pointnet_utils import PointNetEncoder
            self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=6)
        elif self.encoder == "edgeconv":
            from src.models.EdgeConvModel import DgcnnEmbedding
            self.feat = DgcnnEmbedding(args, args.num_category)
        else:
            raise AssertionError("Enter the correct encoder")
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
        if self.encoder == "pointnet":
            x, _, _ = self.feat(pc)
        elif self.encoder == "edgeconv":
            x = self.feat(pc)
        decoder = self.decode(x)
        params = self.out(decoder)
        return params

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, backbone, projection_dim, n_features):
        super(SimCLR, self).__init__()

        self.backbone = backbone
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.backbone.fc = nn.Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x = torch.cat([x.unsqueeze(1), x.unsqueeze(1)], dim=1)
        x = x.view(-1, c, h, w) 

        x = x.cuda(non_blocking=True)

        features = self.contrastive_head(self.backbone(x))
        features = F.normalize(features, dim = 1)
        return features.view(b, 2, -1)

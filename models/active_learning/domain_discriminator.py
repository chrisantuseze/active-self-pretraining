from typing import Optional, Any, Tuple, List, Dict
import numpy as np
import torch.nn as nn
from torch.autograd import Function
import torch

class DomainDiscriminator(nn.Module):
    r"""Domain discriminator model from
    `"Domain-Adversarial Training of Neural Networks" <https://arxiv.org/abs/1505.07818>`_

    Distinguish whether the input features come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.

    Parameters:
        - **in_feature** (int): dimension of the input feature
        - **hidden_size** (int): dimension of the hidden features

    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, 1)`
    """

    def __init__(self, in_feature: int, hidden_size: int):
        super(DomainDiscriminator, self).__init__()
        self.layer1 = nn.Linear(in_feature, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        x = self.relu1(self.bn1(self.layer1(x)))
        x = self.relu2(self.bn2(self.layer2(x)))
        y = self.sigmoid(self.layer3(x))
        return y

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr_mult": 1.}]
    
class DomainClassifier(nn.Module):

    def __init__(self, in_feature: int, bottleneck_dim: Optional[int] = 256):
        super(DomainClassifier, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(in_feature, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        self.domain_discriminator = DomainDiscriminator(in_feature=bottleneck_dim, hidden_size=1024)
        self.grad_reversal = GradientReverseLayer()
        self.coeff = 1.0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        reversed_x = self.grad_reversal(x, self.coeff)

        f = self.bottleneck(reversed_x)
        predictions = self.domain_discriminator(f)
        return predictions#, f

    def get_parameters(self) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
        ] + self.domain_discriminator.get_parameters()
        return params


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        # ctx.coeff = coeff
        # output = input * 1.0
        # return output
        ctx.lambda_ = coeff
        return input.view_as(input)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        # return grad_output.neg() * ctx.coeff, None
        return (grad_output * -ctx.lambda_), None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)
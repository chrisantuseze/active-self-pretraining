from typing import Optional, Any, Tuple, List, Dict
import numpy as np
import torch.nn as nn
from torch.autograd import Function
import torch
import torch.nn.functional as F

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
        return predictions

    def get_parameters(self) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
        ] + self.domain_discriminator.get_parameters()
        return params
    
    def get_loss(self, src_output, trg_output):
        # Ground truth domain labels
        src_domain_labels = torch.zeros_like(src_output) 
        trg_domain_labels = torch.ones_like(trg_output)

        domain_loss = F.binary_cross_entropy(src_output, src_domain_labels) + F.binary_cross_entropy(trg_output, trg_domain_labels)
        conf_loss = F.binary_cross_entropy(src_output, trg_domain_labels) + F.binary_cross_entropy(trg_output, src_domain_labels)

        return domain_loss + 0.5 * conf_loss


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


class VirtualAdversarialLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VirtualAdversarialLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x)[1], dim=1)

        # prepare random unit tensor
        d = torch.randn(x.shape).to(x.device)
        d = _l2_normalize(d)

        # calc adversarial direction
        for _ in range(self.ip):
            d.requires_grad_()

            ad = d * self.xi
            _, pred_hat = model(x + ad)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
            adv_distance.backward()
            d = _l2_normalize(d)
            model.zero_grad()
    
        # calc VAT loss
        r_adv = d * self.eps
        _, pred_hat = model(x + r_adv)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        loss = F.kl_div(logp_hat, pred, reduction='batchmean')

        return loss
    
def entropy_loss(output):
    return - (output * torch.log(output + 3)).sum(dim=1).mean()

def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    temp = torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8

    d = d/temp
    return d

def weight_reg_loss(src_model, trg_model):
    weight_loss = 0
    for src_param, trg_param in zip(src_model.parameters(), trg_model.parameters()):
        weight_loss += ((src_param - trg_param) ** 2).sum()
    return weight_loss
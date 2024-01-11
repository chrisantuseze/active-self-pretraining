from typing import Tuple, Optional, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class DataSetIdx(torch.utils.data.Dataset):
    """ pytorch Dataset that return image index too"""
    def __init__(self, dt):
        self.dt = dt

    def __getitem__(self, index):
        data, target = self.dt[index]
        return data, target, index

    def __len__(self):
        return len(self.dt)


class LabelOptimizer():

    def __init__(self, N, K, lambd, device):
        self.N = N
        self.K = K
        self.lambd = lambd
        self.device = device
        self.P = torch.zeros(N, K).to(device)
        self.Q = torch.zeros(N, K).to(device)
        self.Labels = torch.zeros(N).to(device)
        self.r = 1.
        self.c = 1. * N / K

    def update_P(self, p_t, index):
        # p_batch = p_t / self.N
        self.P[index, :] = p_t

    def update_Labels(self):
        # solve label assignment via sinkhorn-knopp
        self.P = self.P ** self.lambd
        v = (torch.ones(self.K, 1) / self.K).to(self.device)
        err = 1.
        cnt = 0
        while err > 0.1:
            u = self.r / (self.P @ v)
            new_v = self.c / (self.P.T @ u)
            err = torch.sum(torch.abs(new_v / v - 1))
            v = new_v
            cnt += 1
        print(f'error: {err}, step: {cnt}')
        self.Q = u * self.P * v.squeeze()
        # Q = torch.diag(u.squeeze()) @ self.P @ torch.diag(v.squeeze())
        self.Labels = self.Q.argmax(dim=1)


def entropy_loss(p_t):
    # return - (p_t * torch.log(p_t + 1e-5)).sum() / p_t.size(0)
    return (- (p_t * torch.log(p_t + 1e-5)).sum(dim=0)).mean()


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    return d / torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8


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
        d = torch.randn(x[0].shape).to(x[0].device)
        d = _l2_normalize(d)

        # calc adversarial direction
        for _ in range(self.ip):
            d.requires_grad_()

            ad = d * self.xi
            print(x[0].shape, ad.shape)
            _, pred_hat = model(x + [ad])
            logp_hat = F.log_softmax(pred_hat, dim=1)
            adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
            adv_distance.backward()
            d = _l2_normalize(d)
            model.zero_grad()
    
        # calc VAT loss
        r_adv = d * self.eps
        print(r_adv.shape)
        _, pred_hat = model(x + [r_adv])
        logp_hat = F.log_softmax(pred_hat, dim=1)
        loss = F.kl_div(logp_hat, pred, reduction='batchmean')

        return loss


def cross_entropy_ls(pred, label, alpha=0.1):
    ce_loss = F.cross_entropy(pred, label)
    kl_loss = - torch.mean(F.log_softmax(pred, dim=1))
    return (1 - alpha) * ce_loss + alpha * kl_loss


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    return color_distort


def weight_reg_loss(src_model, trg_model):
    weight_loss = 0
    for src_param, trg_param in zip(src_model.parameters(), trg_model.parameters()):
        weight_loss += ((src_param - trg_param) ** 2).sum()
    return weight_loss


def diversity_loss(p_t):
    p_mean = p_t.mean(dim=0)
    return (p_mean * torch.log(p_mean + 1e-5)).sum()

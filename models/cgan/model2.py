import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
import torch.nn.functional as F
import matplotlib.pyplot as plt

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.downsample = downsample

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = F.leaky_relu(self.bn1(out))
        out = F.dropout(out, p=0.2)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(identity)

        out += identity
        out = F.leaky_relu(out)

        return out
    
def make_layer(in_channels, out_channels, blocks=1, stride=1):
    downsample = None
    if (stride != 1) or (in_channels != out_channels):
        downsample = nn.Sequential(conv3x3(in_channels, out_channels, stride=stride))

    layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
    for i in range(1, blocks):
        layers.append(ResidualBlock(out_channels, out_channels))

    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self, z_size, img_size, class_num):
        super().__init__()

        self.z_size = z_size
        self.img_size = img_size
        self.class_num = class_num

        self.label_emb = nn.Embedding(class_num, class_num)
        self.fc = nn.Linear(class_num, class_num * class_num)

        self.rb1 = make_layer(2, 64)
        self.rb2 = make_layer(64, 128)
        self.rb3 = make_layer(128, 256)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

        self.final_fc = nn.Linear(20 * 20, self.img_size * self.img_size)

    def forward(self, z, labels, bs=32):
        # print("g1-z.shape:", z.shape)

        # One-hot vector to embedding vector
        c = self.label_emb(labels)
        # print("\ng1-c.shape:", c.shape)

        c = self.fc(c)
        # print("g2-c.shape:", c.shape)

        # Concat image & label
        x = torch.cat((z, c), dim=1)
        # print("g1-x.shape:", x.shape)

        x = x.view(bs, 2, self.class_num, self.class_num)
        # print("g2-x.shape:", x.shape)

        x = self.rb1(x)
        # print("g3-out.shape:", x.shape)

        x = self.rb2(x)
        # print("g4-out.shape:", x.shape)

        x = self.rb3(x)
        # print("g5-out.shape:", x.shape)

        # Generator out
        out = self.conv_blocks(x)
        # print("g1-out.shape:", out.shape)

        out = out.view(bs, -1)

        out = self.final_fc(out)
        # print("g2-out.shape:", out.shape)

        return out.view(bs, 1, self.img_size, self.img_size)
    

class Discriminator(nn.Module):
    def __init__(self, img_size, class_num, batch_size):
        super().__init__()

        self.label_emb = nn.Embedding(class_num, class_num)
        self.img_size = img_size
        self.batch_size = batch_size
        discriminator_layer_size = [256, 512, 256, 128]

        self.fc = nn.Linear(class_num, self.img_size * self.img_size)

        self.rb1 = make_layer(2, 64)
        self.rb2 = make_layer(64, 128)
        # self.rb3 = make_layer(128, 256)

        self.model = nn.Sequential(
            *self.discriminator_block(128, discriminator_layer_size[0], bn=False),
            *self.discriminator_block(discriminator_layer_size[0], discriminator_layer_size[1]),
            *self.discriminator_block(discriminator_layer_size[1], discriminator_layer_size[2]),
            *self.discriminator_block(discriminator_layer_size[2], discriminator_layer_size[3]),
        )

        self.adv_layer = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid())

    def discriminator_block(self, in_filters, out_filters, bn=True):
        block = [
            nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25)
        ]

        if bn:
            block.append(nn.BatchNorm2d(out_filters, 0.8))
        return block

    def forward(self, x, labels):
        # print("d1-x.shape:", x.shape)

        # One-hot vector to embedding vector
        c = self.label_emb(labels)
        # print("\nd1-c.shape:", c.shape)

        c = self.fc(c)
        # print("d2-c.shape:", c.shape)

        c = c.view(self.batch_size, 1, self.img_size, self.img_size)

        # Concat image & label
        x = torch.cat((x, c), dim=1)
        # print("d3-x.shape:", x.shape)

        x = self.rb1(x)
        # print("d4-out.shape:", x.shape)

        x = self.rb2(x)
        # print("d5-out.shape:", x.shape)

        # x = self.rb3(x)
        # print("d6-out.shape:", x.shape)

        # Discriminator out
        out = self.model(x)
        # print("d1-out.shape:", out.shape)

        out = out.view(out.shape[0], -1)
        # print("d2-out.shape:", out.shape)

        validity = self.adv_layer(out)
        # print("d-validity.shape:", validity.shape)

        return validity
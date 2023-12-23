import torch
from torch import nn
import torchvision
from torchvision.utils import save_image
import torch.optim as optim
from PIL import Image

class Generator(nn.Module):
    def __init__(self, n_channel, z_size, img_size, n_classes, batch_size):
        super(Generator, self).__init__()
        self.z_size = z_size
        self.n_classes = n_classes
        self.img_size = img_size
        self.batch_size = batch_size
        self.n_channel = n_channel

        self.label_emb = nn.Embedding(n_classes, z_size)
        self.label_emb.weight.requires_grad = False
        self.fc = nn.Linear(z_size, z_size)

        self.linear = nn.Linear(2 * z_size, z_size)
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 64*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64*8),
            nn.ReLU(True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(64*8, 64*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(True),

            # # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(64*4, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(64, n_channel, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(n_channel),
            # nn.ReLU(True),

            nn.Tanh()
        )

    def forward(self, z, y):
        # print("g-z.shape", z.shape)
        # print("g-y.shape", y.shape)
        
        y = self.label_emb(y)
        # print("y.shape", y.shape)
        y = self.fc(y)
        # # print("y.shape", y.shape)
        
        z = z.view(-1, self.z_size)
        # print("z.shape", z.shape)

        x = torch.cat([z, y], dim=1)
        # print("x.shape", x.shape)

        x = self.linear(x)
        # print("x.shape", x.shape)

        x = x.unsqueeze(2).unsqueeze(3)
        # print("x.shape", x.shape)

        out = self.main(x)
        # print("gen out.shape", out.shape)

        return out

class Discriminator(nn.Module):
    def __init__(self, n_channel, img_size, n_classes, batch_size):
        super(Discriminator, self).__init__()
        self.n_classes = n_classes
        self.img_size = img_size
        self.batch_size = batch_size
        self.n_channel = n_channel

        self.label_emb = nn.Embedding(n_classes, n_classes * n_classes)
        self.label_emb.weight.requires_grad = False
        self.fc = nn.Linear(n_classes * n_classes, self.img_size * self.img_size)
            
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(n_channel + 1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # # state size. (ndf*4) x 8 x 8
            # nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=0, bias=False),
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(512, n_channel, kernel_size=4, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(n_channel),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.Sigmoid()
        )

    def forward(self, x, y):
        print("x.shape", x.shape)
        print("y.shape", y.shape)

        y = self.label_emb(y)
        print("y.shape", y.shape)
        y = self.fc(y)
        print("y.shape", y.shape)

        y = y.view(self.batch_size, 1, self.img_size, self.img_size)
        # print("y.shape", y.shape)

        inp = torch.cat((x, y), dim=1)
        print("inp.shape", inp.shape)

        out = self.main(inp)
        # print("disc out.shape", out.shape)

        return out.view(self.batch_size, self.n_channel)
import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
from torchsummary import summary
import os
import PIL


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(100, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 3, kernel_size=7, stride=4, padding=2, bias=False),

            nn.Tanh()
        )

    def forward(self, x):
        h = self.gen(x)
        return h


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.gen = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(False),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.gen(x)
        return h


class DCGANG(nn.Module):
    def __init__(self):
        super(DCGANG, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(100, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=5, stride=3, padding=1, bias=False),
            nn.Tanh()
        )

    def weight_init(self):
        for m in self.generator.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0, 0.02)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        h = self.gen(x)
        return h


class DCGAND(nn.Module):
    def __init__(self):
        super(DCGAND, self).__init__()
        self.gen = nn.Sequential(
            nn.Conv2d(3, 64, 5, 3, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(8192, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.gen(x)
        return h


class GanData():
    def __init__(self):
        self.dataset_dir = './anime'
        os.chdir(self.dataset_dir)
        self.img_name = os.listdir()

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        # load image
        img_file = os.path.join(self.dataset_dir, self.img_name[index])
        img = PIL.Image.open(img_file)
        img = img.resize((96, 96))
        img = np.array(img, dtype=np.float32)
        # img = img.reshape((3,64,64))
        img = img.transpose([2, 0, 1])
        img = (img / 255) * 2 - 1
        return img, 1


if __name__ == "__main__":
    summary(DCGANG().cuda(), (100, 1, 1))
    summary(DCGAND().cuda(), (3, 96, 96))
    # GanData()
    exit(0)

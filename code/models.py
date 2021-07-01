import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.modules.pooling import MaxPool2d
import torchvision
import torchvision.transforms as transforms

class StressNet(nn.Module):
    """
    StressNet with/without dropout layers (used for ablation study)
    """
    def __init__(self):
        super(StressNet, self).__init__()

        self.snet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            #nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout2d(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            #nn.BatchNorm2d(512),

            nn.GELU(),
            nn.Dropout2d(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            #nn.BatchNorm2d(1024),
            nn.GELU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, input):
        batch_size = input.size(0)
        return torch.sigmoid(self.snet(input).view(batch_size))


class DynamicStressNet(nn.Module):
    """
    Dynamic Dropout Layer based StressNet decribed in the paper.
    """
    def __init__(self):
        super(DynamicStressNet, self).__init__()

        self.snet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        self.dyn1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            #nn.Sigmoid()
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
        )

        self.gelu_1 = nn.GELU()

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        self.dyn3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            #nn.Sigmoid()
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
        )
        self.gelu_3 = nn.GELU()

        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )

        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, input):
        batch_size = input.size(0)
        output = self.snet(input)
        output_d1 = self.dyn1(output)
        output = self.conv1(output)
        output = self.gelu_1(output * output_d1)
        output = self.conv2(output)
        output_d3 = self.dyn3(output)
        output = self.conv3(output)
        output = self.gelu_3(output * output_d3)
        output = self.conv4(output)
        output = torch.sigmoid(self.out(output).view(batch_size))
        return output
    
    def visualize(self, input):
        batch_size = input.size(0)
        output = self.snet(input)
        output_d1 = self.dyn1(output)
        output = self.conv1(output)
        output = self.gelu_1(output * output_d1)
        output = self.conv2(output)
        output_d3 = self.dyn3(output)
        output = self.conv3(output)
        output = self.gelu_3(output * output_d3)
        output = self.conv4(output)
        output = torch.sigmoid(self.out(output).view(batch_size))
        return output_d1, output_d3




import torch
import torchvision.models as models
from torch import nn


class encoder(nn.Module):

  def __init__(self):
    super().__init__()

    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    self.stem = nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool
    )
    self.layer1 = backbone.layer1
    self.layer2 = backbone.layer2
    self.layer3 = backbone.layer3
    self.layer4 = backbone.layer4

    #ignore last 2 layers of resnet

  def forward(self, x):
    x = self.stem(x)
    c1 = self.layer1(x)
    c2 = self.layer2(c1)
    c3 = self.layer3(c2)
    c4 = self.layer4(c3)
    return c1, c2, c3, c4


class decoder(nn.Module):

  def __init__(self):

    super().__init__()

    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    self.dropout = nn.Dropout2d(0.2)
    self.conv1 = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(256)

    self.conv2 = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(128)

    self.conv3 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.bn3 = nn.BatchNorm2d(64)

    self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.bn4 = nn.BatchNorm2d(32)

    self.conv5 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

  def forward(self, c1, c2, c3, c4):

    x = self.up(c4)
    x = torch.cat([x, c3], dim=1)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.dropout(x)
    x = torch.relu(x)

    x = self.up(x)
    x = torch.cat([x, c2], dim=1)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.dropout(x)
    x = torch.relu(x)

    x = self.up(x)
    x = torch.cat([x, c1], dim=1)
    x = self.conv3(x)
    x = self.bn3(x)
    x = self.dropout(x)
    x = torch.relu(x)

    x = self.up(x)
    x = self.conv4(x)
    x = self.bn4(x)
    x = self.dropout(x)
    x = torch.relu(x)

    x = self.up(x)
    x = self.conv5(x)

    return x

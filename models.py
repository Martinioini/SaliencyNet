import torch
import torchvision.models as models
from torch import nn

#encoder built from a pretrained ResNet18, the last two layers are dropped, and the intermediate are used for skip connections
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

#Decoder: increases size gradually, concatenating the skip connections (U-Net style)
#Uses interpolated upsampling instead of strided convolutions to avoid introducing visual artifacts
#exploits of dropout and batchnorm2d.
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


class discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.dropout = nn.Dropout2d(0.2)
        self.fc_dropout = nn.Dropout(0.2)

        self.conv1_1 = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(3)

        self.conv1_2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)

        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)

        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(64)

        self.conv3_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(64)

        self.FFNN4 = nn.Linear(in_features=64 * 32 * 32, out_features=100)
        self.FFNN5 = nn.Linear(in_features=100, out_features=2)
        self.FFNN6 = nn.Linear(in_features=2, out_features=1)

    def forward(self, image, map):

        x = torch.cat((image, map), dim=1)

        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.dropout(x)
        x = torch.relu(x)

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.dropout(x)
        x = torch.relu(x)

        x = self.pooling(x)

        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.dropout(x)
        x = torch.relu(x)

        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.dropout(x)
        x = torch.relu(x)

        x = self.pooling(x)

        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.dropout(x)
        x = torch.relu(x)

        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.dropout(x)
        x = torch.relu(x)

        x = self.pooling(x)

        x = torch.flatten(x, start_dim=1)
        x = self.FFNN4(x)
        x = self.fc_dropout(x)
        x = torch.tanh(x)

        x = self.FFNN5(x)
        x = self.fc_dropout(x)
        x = torch.tanh(x)

        x = self.FFNN6(x)
        x = torch.sigmoid(x)

        return x

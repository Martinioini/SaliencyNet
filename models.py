import torch
import torchvision.models as models
from torch import nn

# resnet18 encoder. last 2 layers dropped, intermediate features kept for skip connections.
class encoder18(nn.Module):

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

  def forward(self, x):
    x = self.stem(x)
    c1 = self.layer1(x)
    c2 = self.layer2(c1)
    c3 = self.layer3(c2)
    c4 = self.layer4(c3)
    return c1, c2, c3, c4
  
class attention18(nn.Module):
    
    def __init__(self, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.att_c4 = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.gamma_c4 = nn.Parameter(torch.tensor([0.1])) # gating parameter. network can drive this to 0 to switch off attention
        
        self.att_c3 = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.gamma_c3 = nn.Parameter(torch.tensor([0.1]))

    def forward(self, c3, c4):
        
        if not self.use_attention:
            return c3, c4
        
        # Process c4
        B4, C4, H4, W4 = c4.shape
        temp_c4 = c4.flatten(2).transpose(1,2)
        att_output_c4, _ = self.att_c4(temp_c4, temp_c4, temp_c4)
        c4_att = att_output_c4.transpose(1,2).reshape(B4, C4, H4, W4)
        c4_out = c4 + c4_att * self.gamma_c4

        # Process c3
        B3, C3, H3, W3 = c3.shape
        temp_c3 = c3.flatten(2).transpose(1,2)
        att_output_c3, _ = self.att_c3(temp_c3, temp_c3, temp_c3)
        c3_att = att_output_c3.transpose(1,2).reshape(B3, C3, H3, W3)
        c3_out = c3 + c3_att * self.gamma_c3

        return c3_out, c4_out


# bilinear upsampling avoids checkerboard artefacts common in transposed convolutions, which are very visible on saliency maps
class decoder18(nn.Module):

  # use_skip=False consumes only c4 with no concatenation (base model).
  # this is why conv layers have variable in_channels.
  def __init__(self, use_skip: bool = True):

    super().__init__()
    self.use_skip = use_skip

    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    self.dropout = nn.Dropout2d(0.2)
    # 512 (c4) + 256 (c3) = 768
    self.conv1 = nn.Conv2d(in_channels=768 if use_skip else 512, out_channels=256, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(256)

    # 256 + 128 (c2) = 384
    self.conv2 = nn.Conv2d(in_channels=384 if use_skip else 256, out_channels=128, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(128)

    # 128 + 64 (c1) = 192
    self.conv3 = nn.Conv2d(in_channels=192 if use_skip else 128, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.bn3 = nn.BatchNorm2d(64)

    self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.bn4 = nn.BatchNorm2d(32)

    # output is raw logits. Saliency is a probability distribution over the image, so no sigmoid/softmax here.
    self.conv5 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

  def forward(self, c1, c2, c3, c4):

    x = self.up(c4)
    if self.use_skip:
      x = torch.cat([x, c3], dim=1)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.dropout(x)
    x = torch.relu(x)

    x = self.up(x)
    if self.use_skip:
      x = torch.cat([x, c2], dim=1)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.dropout(x)
    x = torch.relu(x)

    x = self.up(x)
    if self.use_skip:
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
  
# resnet50 uses bottleneck blocks, so feature channels are 4x resnet18
class encoder50(nn.Module):

  def __init__(self):
    super().__init__()

    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
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

  def forward(self, x):
    x = self.stem(x)
    c1 = self.layer1(x)
    c2 = self.layer2(c1)
    c3 = self.layer3(c2)
    c4 = self.layer4(c3)
    return c1, c2, c3, c4
  
class attention50(nn.Module):
    
    def __init__(self, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.att_c4 = nn.MultiheadAttention(embed_dim=2048, num_heads=8, batch_first=True)
        self.gamma_c4 = nn.Parameter(torch.tensor([0.1])) # gating parameter

        self.att_c3 = nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
        self.gamma_c3 = nn.Parameter(torch.tensor([0.1]))

    def forward(self, c3, c4):
        if not self.use_attention:
            return c3, c4
        # Process c4
        B4, C4, H4, W4 = c4.shape
        temp_c4 = c4.flatten(2).transpose(1,2)
        att_output_c4, _ = self.att_c4(temp_c4, temp_c4, temp_c4)
        c4_att = att_output_c4.transpose(1,2).reshape(B4, C4, H4, W4)
        c4_out = c4 + c4_att * self.gamma_c4

        # Process c3
        B3, C3, H3, W3 = c3.shape
        temp_c3 = c3.flatten(2).transpose(1,2)
        att_output_c3, _ = self.att_c3(temp_c3, temp_c3, temp_c3)
        c3_att = att_output_c3.transpose(1,2).reshape(B3, C3, H3, W3)
        c3_out = c3 + c3_att * self.gamma_c3

        return c3_out, c4_out


# bilinear upsampling to avoid checkerboard artefacts
class decoder50(nn.Module):

  # use_skip=False consumes only c4 (base model)
  def __init__(self, use_skip: bool = True):

    super().__init__()
    self.use_skip = use_skip

    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    self.dropout = nn.Dropout2d(0.2)
    # 2048 (c4) + 1024 (c3) = 3072
    self.conv1 = nn.Conv2d(in_channels=3072 if use_skip else 2048, out_channels=256, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(256)

    # 256 + 512 (c2) = 768
    self.conv2 = nn.Conv2d(in_channels=768 if use_skip else 256, out_channels=128, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(128)

    # 128 + 256 (c1) = 384
    self.conv3 = nn.Conv2d(in_channels=384 if use_skip else 128, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.bn3 = nn.BatchNorm2d(64)

    self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.bn4 = nn.BatchNorm2d(32)

    self.conv5 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

  def forward(self, c1, c2, c3, c4):

    x = self.up(c4)
    if self.use_skip:
      x = torch.cat([x, c3], dim=1)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.dropout(x)
    x = torch.relu(x)

    x = self.up(x)
    if self.use_skip:
      x = torch.cat([x, c2], dim=1)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.dropout(x)
    x = torch.relu(x)

    x = self.up(x)
    if self.use_skip:
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
  
  

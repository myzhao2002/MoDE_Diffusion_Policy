import torch
import torch.nn as nn
from timm import create_model

class FiLMLayer(nn.Module):
    def __init__(self, num_features, condition_dim):
        super(FiLMLayer, self).__init__()
        self.num_features = num_features
        self.condition_dim = condition_dim
        self.gamma = nn.Linear(condition_dim, num_features)
        self.beta = nn.Linear(condition_dim, num_features)
        
        # Zero initialization
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, condition):
        gamma = self.gamma(condition).unsqueeze(2).unsqueeze(3)
        beta = self.beta(condition).unsqueeze(2).unsqueeze(3)
        x = (1 + gamma) * x + beta  # Using (1 + gamma) to start with identity transform
        return x.contiguous()

class FiLMResNet50Policy(nn.Module):
    def __init__(self, condition_dim):
        super(FiLMResNet50Policy, self).__init__()
        # Load pretrained ResNet50 with weights from ImageNet-1K
        self.resnet = create_model('resnet50', pretrained=True, num_classes=0)
        
        # Add FiLM layers after each residual block
        self.film1 = FiLMLayer(256, condition_dim)
        self.film2 = FiLMLayer(512, condition_dim)
        self.film3 = FiLMLayer(1024, condition_dim)
        self.film4 = FiLMLayer(2048, condition_dim)

    def forward(self, x, condition, return_patches=False):
        if len(condition.shape) == 3:
            condition = condition.squeeze(1)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.act1(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.film1(x, condition)

        x = self.resnet.layer2(x)
        x = self.film2(x, condition)

        x = self.resnet.layer3(x)
        x = self.film3(x, condition)

        x = self.resnet.layer4(x)
        feat = self.film4(x, condition)  # (B, 2048, H, W) language-modulated feature map

        pooled = self.resnet.global_pool(feat).flatten(1)  # (B, 2048) global token
        if return_patches:
            # spatial patch tokens for downstream plan cross-attention:
            # (B, 2048, H, W) -> (B, H*W, 2048). static 224 -> 7x7=49, gripper 112 -> 4x4=16.
            patches = feat.flatten(2).transpose(1, 2).contiguous()
            return pooled, patches

        return pooled  # Return the latent features directly


class FiLMResNet34Policy(nn.Module):
    def __init__(self, condition_dim):
        super(FiLMResNet34Policy, self).__init__()
        # Load pretrained ResNet34 with weights from ImageNet-1K
        self.resnet = create_model('resnet34', pretrained=True, num_classes=0)
        
        # Add FiLM layers after each residual block
        self.film1 = FiLMLayer(64, condition_dim)
        self.film2 = FiLMLayer(128, condition_dim)
        self.film3 = FiLMLayer(256, condition_dim)
        self.film4 = FiLMLayer(512, condition_dim)

    def forward(self, x, condition):
        if len(condition.shape) == 3:
            condition = condition.squeeze(1)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.act1(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.film1(x, condition)

        x = self.resnet.layer2(x)
        x = self.film2(x, condition)

        x = self.resnet.layer3(x)
        x = self.film3(x, condition)

        x = self.resnet.layer4(x)
        x = self.film4(x, condition)

        x = self.resnet.global_pool(x)
        x = x.flatten(1)

        return x  # Return the latent features directly



class FiLMResNet18Policy(nn.Module):
    def __init__(self, condition_dim):
        super(FiLMResNet18Policy, self).__init__()
        # Load pretrained ResNet18 with weights from ImageNet-1K
        self.resnet = create_model('resnet18', pretrained=True, num_classes=0)
        
        # Add FiLM layers after each residual block
        # ResNet18 has the same channel dimensions as ResNet34
        self.film1 = FiLMLayer(64, condition_dim)
        self.film2 = FiLMLayer(128, condition_dim)
        self.film3 = FiLMLayer(256, condition_dim)
        self.film4 = FiLMLayer(512, condition_dim)

    def forward(self, x, condition):
        if len(condition.shape) == 3:
            condition = condition.squeeze(1)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.act1(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.film1(x, condition)

        x = self.resnet.layer2(x)
        x = self.film2(x, condition)

        x = self.resnet.layer3(x)
        x = self.film3(x, condition)

        x = self.resnet.layer4(x)
        x = self.film4(x, condition)

        x = self.resnet.global_pool(x)
        x = x.flatten(1)

        return x  # Return the latent features directly
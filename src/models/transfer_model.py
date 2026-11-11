"""
transfer_model.py - Transfer Learning using Pretrained ResNet-18 for Alzheimer's MRI classification.
"""

import torch
import torch.nn as nn
from torchvision import models

class AlzheimerResNet(nn.Module):
    """
    Transfer Learning model based on Pretrained ResNet-18 with customized classification head.
    Residual connections enable effective feature propagation without gradient degradation.
    """
    def __init__(self, num_classes=4, pretrained=True, freeze_backbone=False):
        super(AlzheimerResNet, self).__init__()

        # Load ResNet-18 with ImageNet weights
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features

        # Replace final fully connected layer with medical classification head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_cam_target_layer(self):
        """Returns the final convolutional layer of ResNet-18 layer4 for Grad-CAM."""
        return self.backbone.layer4[-1].conv2


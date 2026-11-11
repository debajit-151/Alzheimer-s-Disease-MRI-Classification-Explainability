"""
custom_cnn.py - Custom Convolutional Neural Network built from scratch for Alzheimer's MRI classification.
"""

import torch
import torch.nn as nn

class AlzheimerCustomCNN(nn.Module):
    """
    4-Stage Deep Convolutional Neural Network with Batch Normalization,
    Adaptive Pooling, and Dropout for regularization.
    Designed specifically for 2D Brain MRI axial slices.
    """
    def __init__(self, num_classes=4, in_channels=3):
        super(AlzheimerCustomCNN, self).__init__()

        # Conv Block 1: Input -> 32 filters
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv Block 2: 32 -> 64 filters
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv Block 3: 64 -> 128 filters
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.15)
        )

        # Conv Block 4: 128 -> 256 filters (Target feature map for Grad-CAM)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2)
        )

        # Global Average Pooling ensures resolution independence
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier / Fully Connected Layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits

    def get_cam_target_layer(self):
        """Returns the final convolutional layer for Grad-CAM activation mapping."""
        return self.conv_block4[0]

"""
Improved Neural Network Architecture for Shape Recognition
Better handling of small grids and simple shapes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedShapeNet(nn.Module):
    """Improved neural network for structural shape recognition."""

    def __init__(self, num_colors=8, max_instances=10):
        super().__init__()
        self.num_colors = num_colors
        self.max_instances = max_instances

        # Multi-scale feature extraction with less aggressive downsampling
        # Use padding=1 to preserve spatial dimensions better

        # Scale 1: Full resolution
        self.conv1a = nn.Conv2d(num_colors, 32, 3, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1b = nn.BatchNorm2d(32)

        # Scale 2: Features
        self.conv2a = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)

        # Scale 3: Deep features
        self.conv3a = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3b = nn.BatchNorm2d(128)

        # Skip connections
        self.skip1 = nn.Conv2d(32, 128, 1)
        self.skip2 = nn.Conv2d(64, 128, 1)

        # Global context (helps small grids)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_fc = nn.Linear(128, 128)

        # Refinement layer
        self.refine = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_refine = nn.BatchNorm2d(128)

        # Output heads with improved architecture
        self.instance_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, max_instances + 1, 1)
        )

        self.vertex_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

        self.edge_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        # Scale 1
        x1 = F.relu(self.bn1a(self.conv1a(x)))
        x1 = F.relu(self.bn1b(self.conv1b(x1)))

        # Scale 2
        x2 = F.relu(self.bn2a(self.conv2a(x1)))
        x2 = F.relu(self.bn2b(self.conv2b(x2)))

        # Scale 3
        x3 = F.relu(self.bn3a(self.conv3a(x2)))
        x3 = F.relu(self.bn3b(self.conv3b(x3)))

        # Skip connections
        skip1 = self.skip1(x1)
        skip2 = self.skip2(x2)

        # Combine multi-scale features
        features = x3 + skip1 + skip2

        # Global context
        global_ctx = self.global_pool(features)
        global_ctx = self.global_fc(global_ctx.squeeze(-1).squeeze(-1))
        global_ctx = global_ctx.unsqueeze(-1).unsqueeze(-1)
        global_ctx = global_ctx.expand(-1, -1, features.shape[2], features.shape[3])

        # Add global context
        features = features + global_ctx

        # Refinement
        features = F.relu(self.bn_refine(self.refine(features)))

        # Output heads
        instances = self.instance_head(features)
        vertices = torch.sigmoid(self.vertex_head(features))
        edges = torch.sigmoid(self.edge_head(features))

        return {
            'instances': instances,
            'vertices': vertices,
            'edges': edges
        }


# Alias for compatibility
StructuralShapeNet = ImprovedShapeNet

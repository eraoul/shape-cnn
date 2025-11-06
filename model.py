"""
Neural Network Architecture for Structural Shape Recognition
Handles variable-sized grids from 1x1 to 30x30+ with multi-task outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralShapeNet(nn.Module):
    """Neural network for structural shape recognition."""

    def __init__(self, num_colors=8, max_instances=10):
        super().__init__()
        self.num_colors = num_colors
        self.max_instances = max_instances

        # Local feature extraction
        self.conv1 = nn.Conv2d(num_colors, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Global context (helps small grids)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_fc = nn.Linear(128, 128)

        # Output heads
        self.instance_head = nn.Conv2d(128, max_instances + 1, 1)
        self.vertex_head = nn.Conv2d(128, 1, 1)
        self.edge_head = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        # Local path
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        local_features = F.relu(self.bn4(self.conv4(x)))

        # Global context
        global_ctx = self.global_pool(local_features)
        global_ctx = self.global_fc(global_ctx.squeeze(-1).squeeze(-1))
        global_ctx = global_ctx.unsqueeze(-1).unsqueeze(-1)
        global_ctx = global_ctx.expand(-1, -1, local_features.shape[2], local_features.shape[3])

        # Combine
        features = local_features + global_ctx

        # Output heads
        instances = self.instance_head(features)
        vertices = torch.sigmoid(self.vertex_head(features))
        edges = torch.sigmoid(self.edge_head(features))

        return {
            'instances': instances,
            'vertices': vertices,
            'edges': edges
        }

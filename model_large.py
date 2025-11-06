"""
Large Neural Network Architecture for Shape Recognition
Significantly increased capacity for better performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorInvariantConv(nn.Module):
    """
    First convolutional layer with weight sharing across color channels.
    Treats all colors symmetrically while preserving color distinctness.
    """
    def __init__(self, num_colors, out_channels_per_color, kernel_size=3, padding=1):
        super().__init__()
        self.num_colors = num_colors
        self.out_channels_per_color = out_channels_per_color
        # Single kernel applied to each color channel independently
        self.conv = nn.Conv2d(1, out_channels_per_color, kernel_size, padding=padding)

    def forward(self, x):
        # x shape: (batch, num_colors, H, W)
        batch_size, num_colors, H, W = x.shape

        # Process each color with the same kernel
        outputs = []
        for i in range(num_colors):
            # Extract single color channel: (batch, 1, H, W)
            color_channel = x[:, i:i+1, :, :]
            # Apply shared kernel: (batch, out_channels_per_color, H, W)
            out = self.conv(color_channel)
            outputs.append(out)

        # Concatenate along channel dimension
        # Output shape: (batch, num_colors * out_channels_per_color, H, W)
        return torch.cat(outputs, dim=1)


class LargeShapeNet(nn.Module):
    """
    Large architecture with more layers and parameters.
    ~2-3M parameters (vs ~600k in previous version)
    """

    # Shape type constants (simplified: only lines and rectangles)
    SHAPE_CLASSES = {
        'background': 0,
        'line': 1,
        'rectangle': 2,
        'single_cell': 3
    }

    NUM_SHAPE_CLASSES = 4

    def __init__(self, num_colors=8, max_instances=10):
        super().__init__()
        self.num_colors = num_colors
        self.max_instances = max_instances

        # Encoder - deeper with more channels
        # Block 1: Color-invariant first layer + 64 channels
        first_layer_channels = num_colors * 4
        self.conv1a = ColorInvariantConv(num_colors, out_channels_per_color=4, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(first_layer_channels)
        self.conv1b = nn.Conv2d(first_layer_channels, 64, 3, padding=1)
        self.bn1b = nn.BatchNorm2d(64)
        self.conv1c = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1c = nn.BatchNorm2d(64)

        # Block 2: 64 channels
        self.conv2a = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.conv2c = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2c = nn.BatchNorm2d(64)

        # Block 3: 128 channels
        self.conv3a = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.conv3c = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3c = nn.BatchNorm2d(128)

        # Block 4: 256 channels
        self.conv4a = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4a = nn.BatchNorm2d(256)
        self.conv4b = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4b = nn.BatchNorm2d(256)

        # Skip connections
        self.skip1 = nn.Conv2d(64, 256, 1)
        self.skip2 = nn.Conv2d(64, 256, 1)
        self.skip3 = nn.Conv2d(128, 256, 1)

        # Global context with larger dimension
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_fc = nn.Linear(256, 256)

        # Refinement layers
        self.refine1 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_refine1 = nn.BatchNorm2d(256)
        self.refine2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_refine2 = nn.BatchNorm2d(256)

        # Larger output heads
        self.instance_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, max_instances + 1, 1)
        )

        self.vertex_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

        self.edge_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

        # Shape classification head - most important
        self.shape_class_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, self.NUM_SHAPE_CLASSES, 1)
        )

        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        # Block 1
        x1 = F.relu(self.bn1a(self.conv1a(x)))
        x1 = F.relu(self.bn1b(self.conv1b(x1)))
        x1 = F.relu(self.bn1c(self.conv1c(x1)))

        # Block 2
        x2 = F.relu(self.bn2a(self.conv2a(x1)))
        x2 = F.relu(self.bn2b(self.conv2b(x2)))
        x2 = F.relu(self.bn2c(self.conv2c(x2)))

        # Block 3
        x3 = F.relu(self.bn3a(self.conv3a(x2)))
        x3 = F.relu(self.bn3b(self.conv3b(x3)))
        x3 = F.relu(self.bn3c(self.conv3c(x3)))

        # Block 4
        x4 = F.relu(self.bn4a(self.conv4a(x3)))
        x4 = F.relu(self.bn4b(self.conv4b(x4)))

        # Skip connections
        skip1 = self.skip1(x1)
        skip2 = self.skip2(x2)
        skip3 = self.skip3(x3)

        # Combine multi-scale features
        features = x4 + skip1 + skip2 + skip3

        # Global context
        global_ctx = self.global_pool(features)
        global_ctx = self.global_fc(global_ctx.squeeze(-1).squeeze(-1))
        global_ctx = global_ctx.unsqueeze(-1).unsqueeze(-1)
        global_ctx = global_ctx.expand(-1, -1, features.shape[2], features.shape[3])

        features = features + global_ctx

        # Refinement with dropout
        features = F.relu(self.bn_refine1(self.refine1(features)))
        features = self.dropout(features)
        features = F.relu(self.bn_refine2(self.refine2(features)))

        # Output heads
        instances = self.instance_head(features)
        vertices = torch.sigmoid(self.vertex_head(features))
        edges = torch.sigmoid(self.edge_head(features))
        shape_classes = self.shape_class_head(features)

        return {
            'instances': instances,
            'vertices': vertices,
            'edges': edges,
            'shape_classes': shape_classes
        }

    @staticmethod
    def get_shape_name(class_id):
        """Convert class ID to shape name."""
        id_to_name = {v: k for k, v in LargeShapeNet.SHAPE_CLASSES.items()}
        return id_to_name.get(class_id, 'unknown')

    @staticmethod
    def get_shape_id(shape_name):
        """Convert shape name to class ID."""
        return LargeShapeNet.SHAPE_CLASSES.get(shape_name, 0)

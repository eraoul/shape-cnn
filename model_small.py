"""
Neural Network with Learned Shape Classification
Instead of heuristic-based classification, the network learns to classify shapes
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


class SmallShapeNet(nn.Module):
    """
    Small shape recognition network with learned classification head.
    ~600k parameters with 3 convolutional blocks.

    Outputs:
    - Instance segmentation (which pixels belong to which shape)
    - Vertex detection (where vertices are)
    - Edge detection (where edges are)
    - Shape classification (what type of shape each pixel belongs to)
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

        # Multi-scale feature extraction
        # Color-invariant first layer: 4 features per color
        first_layer_channels = num_colors * 4
        self.conv1a = ColorInvariantConv(num_colors, out_channels_per_color=4, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(first_layer_channels)
        self.conv1b = nn.Conv2d(first_layer_channels, 64, 3, padding=1)
        self.bn1b = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.2)

        self.conv2a = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(0.2)

        self.conv3a = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(0.2)

        # Skip connections
        self.skip1 = nn.Conv2d(64, 128, 1)
        self.skip2 = nn.Conv2d(64, 128, 1)

        # Global context
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_fc = nn.Linear(128, 128)

        # Refinement
        self.refine = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_refine = nn.BatchNorm2d(128)
        self.dropout_refine = nn.Dropout2d(0.3)

        # Output heads
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

        # NEW: Shape classification head
        self.shape_class_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, self.NUM_SHAPE_CLASSES, 1)
        )

    def forward(self, x):
        # Scale 1
        x1 = F.relu(self.bn1a(self.conv1a(x)))
        x1 = F.relu(self.bn1b(self.conv1b(x1)))
        x1 = self.dropout1(x1)

        # Scale 2
        x2 = F.relu(self.bn2a(self.conv2a(x1)))
        x2 = F.relu(self.bn2b(self.conv2b(x2)))
        x2 = self.dropout2(x2)

        # Scale 3
        x3 = F.relu(self.bn3a(self.conv3a(x2)))
        x3 = F.relu(self.bn3b(self.conv3b(x3)))
        x3 = self.dropout3(x3)

        # Skip connections
        skip1 = self.skip1(x1)
        skip2 = self.skip2(x2)

        # Combine
        features = x3 + skip1 + skip2

        # Global context
        global_ctx = self.global_pool(features)
        global_ctx = self.global_fc(global_ctx.squeeze(-1).squeeze(-1))
        global_ctx = global_ctx.unsqueeze(-1).unsqueeze(-1)
        global_ctx = global_ctx.expand(-1, -1, features.shape[2], features.shape[3])

        features = features + global_ctx

        # Refinement
        features = F.relu(self.bn_refine(self.refine(features)))
        features = self.dropout_refine(features)

        # Output heads
        instances = self.instance_head(features)
        vertices = torch.sigmoid(self.vertex_head(features))
        edges = torch.sigmoid(self.edge_head(features))
        shape_classes = self.shape_class_head(features)  # Logits for cross-entropy

        return {
            'instances': instances,
            'vertices': vertices,
            'edges': edges,
            'shape_classes': shape_classes
        }

    @staticmethod
    def get_shape_name(class_id):
        """Convert class ID to shape name."""
        id_to_name = {v: k for k, v in SmallShapeNet.SHAPE_CLASSES.items()}
        return id_to_name.get(class_id, 'unknown')

    @staticmethod
    def get_shape_id(shape_name):
        """Convert shape name to class ID."""
        return SmallShapeNet.SHAPE_CLASSES.get(shape_name, 0)


# Aliases for backward compatibility
ShapeNetWithClassification = SmallShapeNet
StructuralShapeNet = SmallShapeNet

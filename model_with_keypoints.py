"""
Shape Recognition Model with Keypoint Regression
Direct prediction of vertex coordinates for each instance
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


class ShapeNetWithKeypoints(nn.Module):
    """
    Shape recognition network with keypoint regression.

    Key innovation: Directly predicts (y, x) coordinates for vertices
    instead of using a heatmap + clustering approach.

    Outputs:
    - Instance segmentation (which pixels belong to which shape)
    - Shape classification (per-pixel shape type)
    - Keypoints (per-instance vertex coordinates)
    """

    # Shape type constants
    SHAPE_CLASSES = {
        'background': 0,
        'line': 1,
        'rectangle': 2,
        'single_cell': 3
    }

    # Number of keypoints per shape type
    MAX_KEYPOINTS_PER_SHAPE = {
        'background': 0,
        'single_cell': 1,  # Center point
        'line': 2,         # Two endpoints
        'rectangle': 4,    # Four corners
    }

    MAX_KEYPOINTS = 4  # Maximum across all shape types

    NUM_SHAPE_CLASSES = 4

    def __init__(self, num_colors=10, max_instances=10):
        super().__init__()
        self.num_colors = num_colors
        self.max_instances = max_instances

        # Encoder: Multi-scale feature extraction
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

        # Dense output heads (per-pixel predictions)
        self.instance_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, max_instances + 1, 1)
        )

        self.shape_class_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, self.NUM_SHAPE_CLASSES, 1)
        )

        # NEW: Keypoint regression head (per-instance predictions)
        # Uses global pooling per instance + spatial context (centroid, bbox)
        # Input: 128 (features) + 2 (centroid) + 4 (bbox) = 134
        # Output: [batch, max_instances, max_keypoints, 3]
        #   where 3 = (y_coord, x_coord, validity)
        self.keypoint_head = nn.Sequential(
            nn.Linear(134, 128),  # 128 features + 2 centroid + 4 bbox
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.MAX_KEYPOINTS * 3)  # 4 keypoints * (y, x, valid)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        H, W = x.shape[2], x.shape[3]

        # Encoder
        x1 = F.relu(self.bn1a(self.conv1a(x)))
        x1 = F.relu(self.bn1b(self.conv1b(x1)))
        x1 = self.dropout1(x1)

        x2 = F.relu(self.bn2a(self.conv2a(x1)))
        x2 = F.relu(self.bn2b(self.conv2b(x2)))
        x2 = self.dropout2(x2)

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

        # Dense predictions
        instances = self.instance_head(features)  # [batch, max_instances+1, H, W]
        shape_classes = self.shape_class_head(features)  # [batch, num_classes, H, W]

        # Keypoint predictions: pool features per instance
        keypoints = self._predict_keypoints(features, instances)

        return {
            'instances': instances,
            'shape_classes': shape_classes,
            'keypoints': keypoints  # [batch, max_instances, max_keypoints, 3]
        }

    def _predict_keypoints(self, features, instance_logits):
        """
        Predict keypoints for each instance by pooling features per instance.

        Args:
            features: [batch, 128, H, W] feature maps
            instance_logits: [batch, max_instances+1, H, W] instance segmentation logits

        Returns:
            keypoints: [batch, max_instances, max_keypoints, 3]
                where 3 = (y_normalized, x_normalized, validity)
        """
        batch_size = features.shape[0]
        H, W = features.shape[2], features.shape[3]

        # Get instance masks from logits
        instance_probs = F.softmax(instance_logits, dim=1)  # [batch, max_instances+1, H, W]

        keypoints_list = []

        for inst_id in range(1, self.max_instances + 1):  # Skip background (0)
            # Get mask for this instance: [batch, H, W]
            inst_mask = instance_probs[:, inst_id, :, :]

            # Threshold and normalize mask
            inst_mask = (inst_mask > 0.1).float()  # Binary mask

            # Pool features weighted by instance mask
            # [batch, 128, H, W] * [batch, 1, H, W] -> [batch, 128, H, W]
            masked_features = features * inst_mask.unsqueeze(1)

            # Average pooling over spatial dimensions
            # Sum over H, W and divide by mask sum
            pooled = masked_features.sum(dim=[2, 3])  # [batch, 128]
            mask_sum = inst_mask.sum(dim=[1, 2]).unsqueeze(1) + 1e-6  # [batch, 1]
            pooled = pooled / mask_sum  # [batch, 128]

            # Compute spatial context from mask
            # Create coordinate grids
            y_coords = torch.arange(H, device=features.device).float().view(1, H, 1).expand(batch_size, H, W)
            x_coords = torch.arange(W, device=features.device).float().view(1, 1, W).expand(batch_size, H, W)

            # Normalize coordinates to [0, 1]
            y_coords_norm = y_coords / (H - 1) if H > 1 else y_coords
            x_coords_norm = x_coords / (W - 1) if W > 1 else x_coords

            # Compute weighted centroid
            centroid_y = (y_coords_norm * inst_mask).sum(dim=[1, 2]) / mask_sum.squeeze(1)  # [batch]
            centroid_x = (x_coords_norm * inst_mask).sum(dim=[1, 2]) / mask_sum.squeeze(1)  # [batch]

            # Compute bounding box
            # Find min/max for each batch element
            bbox_list = []
            for b in range(batch_size):
                mask_b = inst_mask[b]
                if mask_b.sum() > 0:
                    y_indices, x_indices = torch.where(mask_b > 0.5)
                    y_min = y_indices.min().float() / (H - 1) if H > 1 else y_indices.min().float()
                    y_max = y_indices.max().float() / (H - 1) if H > 1 else y_indices.max().float()
                    x_min = x_indices.min().float() / (W - 1) if W > 1 else x_indices.min().float()
                    x_max = x_indices.max().float() / (W - 1) if W > 1 else x_indices.max().float()
                else:
                    # No mask - use defaults
                    y_min, y_max, x_min, x_max = 0.5, 0.5, 0.5, 0.5
                bbox_list.append(torch.tensor([y_min, y_max, x_min, x_max], device=features.device))

            bbox = torch.stack(bbox_list)  # [batch, 4]

            # Concatenate: pooled features + centroid + bbox
            centroid = torch.stack([centroid_y, centroid_x], dim=1)  # [batch, 2]
            spatial_context = torch.cat([pooled, centroid, bbox], dim=1)  # [batch, 134]

            # Predict keypoints from features + spatial context
            kp_pred = self.keypoint_head(spatial_context)  # [batch, max_keypoints * 3]
            kp_pred = kp_pred.view(batch_size, self.MAX_KEYPOINTS, 3)  # [batch, max_keypoints, 3]

            keypoints_list.append(kp_pred)

        # Stack all instances: [batch, max_instances, max_keypoints, 3]
        keypoints = torch.stack(keypoints_list, dim=1)

        # Apply sigmoid to coordinates (normalize to [0, 1])
        keypoints[:, :, :, 0] = torch.sigmoid(keypoints[:, :, :, 0])  # y
        keypoints[:, :, :, 1] = torch.sigmoid(keypoints[:, :, :, 1])  # x
        keypoints[:, :, :, 2] = torch.sigmoid(keypoints[:, :, :, 2])  # validity

        return keypoints

    @staticmethod
    def get_shape_name(class_id):
        """Convert class ID to shape name."""
        id_to_name = {v: k for k, v in ShapeNetWithKeypoints.SHAPE_CLASSES.items()}
        return id_to_name.get(class_id, 'unknown')

    @staticmethod
    def get_shape_id(shape_name):
        """Convert shape name to class ID."""
        return ShapeNetWithKeypoints.SHAPE_CLASSES.get(shape_name, 0)

    @staticmethod
    def get_num_keypoints(shape_type):
        """Get expected number of keypoints for a shape type."""
        return ShapeNetWithKeypoints.MAX_KEYPOINTS_PER_SHAPE.get(shape_type, 0)

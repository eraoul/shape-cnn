"""
Simplified U-Net architecture for shape detection.

Key improvements over previous models:
- Standard U-Net encoder-decoder structure
- No restrictive color-invariant layers
- Cleaner multi-task output heads
- Better skip connections
- Proper normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two conv layers with BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ShapeUNet(nn.Module):
    """
    U-Net for multi-task shape detection.

    Outputs:
    - Instance segmentation: which shape instance each pixel belongs to
    - Vertex detection: probability of pixel being a vertex/corner
    - Edge detection: probability of pixel being an edge
    - Shape classification: shape type for each pixel
    """

    def __init__(
        self,
        num_colors: int = 10,
        max_instances: int = 10,
        num_shape_classes: int = 6,
        bilinear: bool = True
    ):
        super().__init__()
        self.num_colors = num_colors
        self.max_instances = max_instances
        self.num_shape_classes = num_shape_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(num_colors, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Output heads
        # Instance segmentation: max_instances + 1 (background)
        self.instance_head = nn.Conv2d(64, max_instances + 1, kernel_size=1)

        # Vertex detection: binary heatmap
        self.vertex_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Edge detection: binary heatmap
        self.edge_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Shape classification: per-pixel class prediction
        self.shape_class_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_shape_classes, kernel_size=1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, num_colors, H, W) - one-hot encoded grid

        Returns:
            dict with keys:
            - instances: (batch, max_instances+1, H, W) - logits
            - vertices: (batch, 1, H, W) - probabilities [0, 1]
            - edges: (batch, 1, H, W) - probabilities [0, 1]
            - shape_classes: (batch, num_shape_classes, H, W) - logits
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output heads
        instances = self.instance_head(x)
        vertices = self.vertex_head(x)
        edges = self.edge_head(x)
        shape_classes = self.shape_class_head(x)

        return {
            'instances': instances,
            'vertices': vertices,
            'edges': edges,
            'shape_classes': shape_classes,
        }


class CompactShapeUNet(nn.Module):
    """
    Smaller U-Net variant for faster training.

    Differences from ShapeUNet:
    - Fewer channels (32-64-128-256 instead of 64-128-256-512-1024)
    - Only 3 down/up blocks instead of 4
    - ~1/4 the parameters
    """

    def __init__(
        self,
        num_colors: int = 10,
        max_instances: int = 10,
        num_shape_classes: int = 6,
        bilinear: bool = True
    ):
        super().__init__()
        self.num_colors = num_colors
        self.max_instances = max_instances
        self.num_shape_classes = num_shape_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(num_colors, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down3 = Down(128, 256 // factor)

        # Decoder
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32, bilinear)

        # Output heads (same as full model)
        self.instance_head = nn.Conv2d(32, max_instances + 1, kernel_size=1)

        self.vertex_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.edge_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.shape_class_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_shape_classes, kernel_size=1)
        )

    def forward(self, x):
        """Same interface as ShapeUNet."""
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder with skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # Output heads
        instances = self.instance_head(x)
        vertices = self.vertex_head(x)
        edges = self.edge_head(x)
        shape_classes = self.shape_class_head(x)

        return {
            'instances': instances,
            'vertices': vertices,
            'edges': edges,
            'shape_classes': shape_classes,
        }


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model
    print("Testing ShapeUNet...")

    # Full model
    model_full = ShapeUNet(num_colors=10, max_instances=10, num_shape_classes=6)
    print(f"Full U-Net parameters: {count_parameters(model_full):,}")

    # Compact model
    model_compact = CompactShapeUNet(num_colors=10, max_instances=10, num_shape_classes=6)
    print(f"Compact U-Net parameters: {count_parameters(model_compact):,}")

    # Test forward pass
    batch_size = 4
    h, w = 24, 24
    x = torch.randn(batch_size, 10, h, w)

    with torch.no_grad():
        outputs = model_compact(x)

    print(f"\nOutput shapes:")
    print(f"  instances: {outputs['instances'].shape}")
    print(f"  vertices: {outputs['vertices'].shape}")
    print(f"  edges: {outputs['edges'].shape}")
    print(f"  shape_classes: {outputs['shape_classes'].shape}")

    print("\nModel test passed!")

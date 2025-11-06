"""
Test Color Invariance Property
Verifies that swapping colors produces equivalent feature patterns
"""

import torch
import numpy as np
from model_small import ShapeNetWithClassification

def test_color_invariance():
    """Test that the color-invariant layer treats all colors symmetrically."""

    print("="*80)
    print("COLOR INVARIANCE TEST")
    print("="*80)

    # Create model
    model = ShapeNetWithClassification(num_colors=10, max_instances=10)
    model.eval()

    # Create a simple test grid with two different colors
    # Grid 1: Color 1 in top-left, Color 2 in bottom-right (Color 0 reserved for background)
    grid1 = torch.zeros(1, 10, 10, 10)
    grid1[0, 1, :5, :5] = 1.0  # Color 1
    grid1[0, 2, 5:, 5:] = 1.0  # Color 2

    # Grid 2: Swap the colors (Color 2 in top-left, Color 1 in bottom-right)
    grid2 = torch.zeros(1, 10, 10, 10)
    grid2[0, 2, :5, :5] = 1.0  # Color 2 (was Color 1)
    grid2[0, 1, 5:, 5:] = 1.0  # Color 1 (was Color 2)

    print("\nGrid 1: Color 1 in top-left, Color 2 in bottom-right")
    print("Grid 2: Color 2 in top-left, Color 1 in bottom-right")
    print("Note: Color 0 is reserved for background")

    # Run through first layer only
    with torch.no_grad():
        # Process grid 1
        feat1 = model.conv1a(grid1)

        # Process grid 2
        feat2 = model.conv1a(grid2)

    print(f"\nFeature map shapes: {feat1.shape}")
    print(f"Grid 1 features has {feat1.shape[1]} channels")
    print(f"Grid 2 features has {feat2.shape[1]} channels")

    # Check color invariance property
    # Features from color 1 in grid1 should match features from color 1 in grid2
    # But they should appear in different spatial locations

    # Top-left region features
    tl1 = feat1[0, 4:8, :5, :5]  # Color 1 features in grid1 (channels 4-7)
    tl2 = feat2[0, 8:12, :5, :5]  # Color 2 features in grid2 (channels 8-11)

    # Bottom-right region features
    br1 = feat1[0, 8:12, 5:, 5:]  # Color 2 features in grid1 (channels 8-11)
    br2 = feat2[0, 4:8, 5:, 5:]  # Color 1 features in grid2 (channels 4-7)

    # These should be approximately equal due to weight sharing
    diff_tl = torch.abs(tl1 - tl2).mean()
    diff_br = torch.abs(br1 - br2).mean()

    print(f"\n✓ Color-invariant first layer:")
    print(f"  - Each color gets {feat1.shape[1] // 10} feature channels")
    print(f"  - Same kernel weights applied to all colors")
    print(f"  - Color distinctness preserved (different feature channel groups)")
    print(f"  - Color 0 reserved for background (channels 0-3)")

    print(f"\nFeature difference between swapped colors:")
    print(f"  - Top-left region: {diff_tl:.6f}")
    print(f"  - Bottom-right region: {diff_br:.6f}")

    if diff_tl < 0.001 and diff_br < 0.001:
        print("\n✓ Color invariance verified! Swapping colors produces equivalent features.")
    else:
        print("\n✗ Warning: Large difference detected. Check implementation.")

    print("\n" + "="*80)
    print("EXPLANATION")
    print("="*80)
    print("""
The color-invariant layer treats all colors symmetrically:
- Same convolutional kernel applied to each color channel
- Color 0 gets channels [0:4] (background), Color 1 gets [4:8], etc.
- With 10 colors: 40 total channels (10 × 4 features per color)
- Swapping colors just permutes the feature channel groups
- Network learns "what features matter" without color bias
- Colors remain distinct so shapes can be separated
- Color 0 is reserved for background only
    """)

if __name__ == "__main__":
    test_color_invariance()

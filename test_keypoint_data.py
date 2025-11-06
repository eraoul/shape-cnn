"""
Test that keypoint data generation is correct
"""

import numpy as np
from data_generation_with_keypoints import ShapeGeneratorWithKeypoints

def test_keypoint_targets():
    """Test that keypoint targets are generated correctly."""

    generator = ShapeGeneratorWithKeypoints(num_colors=10, min_size=10, max_size=15, max_instances=10)

    print("Testing keypoint data generation...")
    print("="*80)

    for i in range(5):
        sample = generator.generate_sample()
        h, w = sample['grid_size']
        keypoint_targets = sample['keypoint_targets']  # [max_instances, max_keypoints, 3]

        print(f"\nSample {i+1}: Grid size {h}x{w}")

        if 'shapes' in sample['metadata']:
            for shape_idx, shape_meta in enumerate(sample['metadata']['shapes']):
                print(f"\n  Shape {shape_idx+1}: {shape_meta['type']}")
                print(f"    Ground Truth Vertices:")
                for v_idx, (y, x) in enumerate(shape_meta['vertices']):
                    print(f"      V{v_idx+1}: ({y:.1f}, {x:.1f})")

                print(f"    Keypoint Targets (normalized):")
                kp_targets = keypoint_targets[shape_idx]
                for kp_idx in range(len(kp_targets)):
                    y_norm, x_norm, valid = kp_targets[kp_idx]
                    if valid > 0.5:
                        # Denormalize to check
                        y_pixel = y_norm * h
                        x_pixel = x_norm * w
                        print(f"      KP{kp_idx+1}: norm=({y_norm:.3f}, {x_norm:.3f}) -> pixel=({y_pixel:.1f}, {x_pixel:.1f})")

                        # Check if it matches ground truth
                        if kp_idx < len(shape_meta['vertices']):
                            gt_y, gt_x = shape_meta['vertices'][kp_idx]
                            error = np.sqrt((y_pixel - gt_y)**2 + (x_pixel - gt_x)**2)
                            if error > 0.1:
                                print(f"        WARNING: Error {error:.2f} pixels from GT!")
                            else:
                                print(f"        ✓ Matches GT")
        else:
            # Single shape
            shape_type = sample['metadata']['type']
            print(f"\n  Single {shape_type}")
            kp_targets = keypoint_targets[0]
            for kp_idx in range(len(kp_targets)):
                y_norm, x_norm, valid = kp_targets[kp_idx]
                if valid > 0.5:
                    y_pixel = y_norm * h
                    x_pixel = x_norm * w
                    print(f"    KP{kp_idx+1}: norm=({y_norm:.3f}, {x_norm:.3f}) -> pixel=({y_pixel:.1f}, {x_pixel:.1f})")

    print("\n" + "="*80)
    print("Test complete!")

if __name__ == "__main__":
    test_keypoint_targets()

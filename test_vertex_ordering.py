"""
Test script to verify vertex ordering logic
"""

import numpy as np
import torch
from data_generation_with_types import ShapeGeneratorWithTypes
from model_large import LargeShapeNet
from model_small import ShapeNetWithClassification as SmallShapeNet
from inference_with_classification import ShapeInferenceWithClassification


def test_vertex_ordering():
    """Test that vertex ordering is working correctly."""

    # Load model
    print("Loading model...")
    model = SmallShapeNet(num_colors=10, max_instances=10)
    model.load_state_dict(torch.load('best_model_small.pth'))

    # Create inference engine
    inference = ShapeInferenceWithClassification(model)
    print(f"Using device: {inference.device}")

    # Generate a test sample
    print("\nGenerating test sample...")
    generator = ShapeGeneratorWithTypes(num_colors=10, min_size=10, max_size=15)
    sample = generator.generate_sample()

    # Run inference
    print("Running inference...")
    prediction = inference.predict(sample['grid'])

    # Display results
    print("\n" + "="*80)
    print("VERTEX ORDERING TEST RESULTS")
    print("="*80)
    print(f"Grid size: {prediction['grid_size']}")
    print(f"Number of detected objects: {len(prediction['objects'])}")

    for i, obj in enumerate(prediction['objects']):
        print(f"\nObject {i+1}:")
        print(f"  Type: {obj['type']}")
        print(f"  Color: {obj['color']}")
        print(f"  Number of vertices: {len(obj['vertices'])}")

        if len(obj['vertices']) > 0:
            print(f"  Vertices (ordered):")
            for vi, (vy, vx) in enumerate(obj['vertices']):
                print(f"    V{vi+1}: ({vy:.2f}, {vx:.2f})")

            # Check ordering for rectangles
            if obj['type'] == 'rectangle' and len(obj['vertices']) == 4:
                vertices = np.array(obj['vertices'])
                centroid = vertices.mean(axis=0)

                # Verify vertices are ordered by angle
                angles = np.arctan2(vertices[:, 0] - centroid[0],
                                   vertices[:, 1] - centroid[1])

                is_ordered = np.all(angles[:-1] <= angles[1:])
                print(f"  Vertices correctly ordered: {is_ordered}")
                print(f"  Angles from centroid: {np.degrees(angles)}")

            # Check ordering for lines
            elif obj['type'] == 'line' and len(obj['vertices']) == 2:
                v1, v2 = obj['vertices']
                dist = np.linalg.norm(np.array(v1) - np.array(v2))
                print(f"  Distance between endpoints: {dist:.2f}")

    print("\n" + "="*80)
    print("Test complete!")


if __name__ == "__main__":
    test_vertex_ordering()

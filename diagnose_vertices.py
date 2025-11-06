"""
Diagnostic script to visualize vertex prediction issues
Compares ground truth vertices with predicted vertex heatmap
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from data_generation_with_types import ShapeGeneratorWithTypes
from model_small import ShapeNetWithClassification as SmallShapeNet
from inference_with_classification import ShapeInferenceWithClassification


def visualize_vertex_predictions(num_samples=5):
    """Visualize vertex predictions vs ground truth."""

    # Load model
    print("Loading model...")
    model = SmallShapeNet(num_colors=10, max_instances=10)
    model.load_state_dict(torch.load('best_model_small.pth'))
    inference = ShapeInferenceWithClassification(model)
    print(f"Using device: {inference.device}")

    # Generate samples
    print("\nGenerating samples...")
    generator = ShapeGeneratorWithTypes(num_colors=10, min_size=8, max_size=15)

    for sample_idx in range(num_samples):
        sample = generator.generate_sample()
        grid = sample['grid']
        gt_vertex_map = sample['vertex_map']
        h, w = grid.shape[:2]

        # Run inference
        prediction = inference.predict(grid)

        # Get predictions
        pred_vertex_map = prediction['raw_outputs']['vertex_map']
        instance_map = prediction['raw_outputs']['instance_map']

        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Sample {sample_idx + 1}: Vertex Prediction Diagnosis',
                     fontsize=14, fontweight='bold')

        # Input grid
        input_rgb = grid.argmax(axis=2)
        axes[0, 0].imshow(input_rgb, cmap='tab10', interpolation='nearest', vmin=0, vmax=9)
        axes[0, 0].set_title('Input Grid', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # Ground truth vertices
        axes[0, 1].imshow(gt_vertex_map, cmap='hot', interpolation='nearest')
        axes[0, 1].set_title('Ground Truth Vertices', fontsize=12, fontweight='bold')

        # Mark GT vertices with crosses
        gt_vertices = np.argwhere(gt_vertex_map > 0.5)
        for vy, vx in gt_vertices:
            axes[0, 1].plot(vx, vy, 'gx', markersize=15, markeredgewidth=3)
        axes[0, 1].set_xlim(-0.5, w - 0.5)
        axes[0, 1].set_ylim(h - 0.5, -0.5)
        axes[0, 1].axis('off')

        # Predicted vertex heatmap
        axes[0, 2].imshow(pred_vertex_map, cmap='hot', interpolation='nearest')
        axes[0, 2].set_title('Predicted Vertex Heatmap', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')

        # Predicted vertices (thresholded at 0.3)
        axes[1, 0].imshow(input_rgb, cmap='tab10', interpolation='nearest', vmin=0, vmax=9, alpha=0.5)
        axes[1, 0].imshow(pred_vertex_map > 0.3, cmap='hot', interpolation='nearest', alpha=0.5)
        axes[1, 0].set_title('Predicted Vertices (threshold=0.3)', fontsize=12, fontweight='bold')

        # Mark extracted vertices
        for obj in prediction['objects']:
            for vy, vx in obj['vertices']:
                axes[1, 0].plot(vx, vy, 'c*', markersize=20, markeredgewidth=2,
                              markeredgecolor='white')
        axes[1, 0].axis('off')

        # Comparison: GT vs Predicted
        axes[1, 1].imshow(input_rgb, cmap='tab10', interpolation='nearest', vmin=0, vmax=9, alpha=0.3)

        # GT vertices in green
        for vy, vx in gt_vertices:
            axes[1, 1].plot(vx, vy, 'go', markersize=12, markerfacecolor='none',
                          markeredgewidth=3, label='GT' if vy == gt_vertices[0][0] and vx == gt_vertices[0][1] else '')

        # Predicted vertices in cyan
        for obj in prediction['objects']:
            for vy, vx in obj['vertices']:
                axes[1, 1].plot(vx, vy, 'c^', markersize=12, markerfacecolor='none',
                              markeredgewidth=3, label='Pred' if obj == prediction['objects'][0] and (vy, vx) == obj['vertices'][0] else '')

        axes[1, 1].set_title('GT (green circles) vs Predicted (cyan triangles)',
                           fontsize=12, fontweight='bold')
        axes[1, 1].legend(loc='upper right')
        axes[1, 1].set_xlim(-0.5, w - 0.5)
        axes[1, 1].set_ylim(h - 0.5, -0.5)
        axes[1, 1].axis('off')

        # Statistics
        stats_text = f"Ground Truth Metadata:\n"
        if 'shapes' in sample['metadata']:
            for i, shape_meta in enumerate(sample['metadata']['shapes']):
                stats_text += f"  Shape {i+1}: {shape_meta['type']}\n"
                stats_text += f"    Vertices: {len(shape_meta['vertices'])}\n"
        else:
            stats_text += f"  Type: {sample['metadata'].get('type', 'unknown')}\n"

        stats_text += f"\nPredicted Objects: {len(prediction['objects'])}\n"
        for i, obj in enumerate(prediction['objects']):
            stats_text += f"  Object {i+1}: {obj['type']}\n"
            stats_text += f"    Detected vertices: {len(obj['vertices'])}\n"

        # Max heatmap values
        stats_text += f"\nVertex Heatmap Stats:\n"
        stats_text += f"  Max value: {pred_vertex_map.max():.3f}\n"
        stats_text += f"  Pixels > 0.3: {(pred_vertex_map > 0.3).sum()}\n"
        stats_text += f"  Pixels > 0.5: {(pred_vertex_map > 0.5).sum()}\n"
        stats_text += f"  GT vertices: {len(gt_vertices)}\n"

        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                       family='monospace')
        axes[1, 2].set_title('Statistics', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

        print(f"\nSample {sample_idx + 1} displayed. Close window to continue...")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    visualize_vertex_predictions(n)

"""
Diagnose Shape Classification Issues
Shows what's going wrong with vertex detection and shape classification
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from data_generation import ShapeDataset
from model import StructuralShapeNet
from inference import ShapeInference


def add_gridlines(ax, h, w):
    """Add gridlines."""
    for x in np.arange(0.5, w, 1):
        ax.axvline(x, color='white', linewidth=0.5, alpha=0.3)
    for y in np.arange(0.5, h, 1):
        ax.axhline(y, color='white', linewidth=0.5, alpha=0.3)


def diagnose_sample(dataset_path, model_path, sample_idx):
    """
    Deep dive into a specific sample to see what's going wrong.
    """
    # Load
    dataset = ShapeDataset(num_samples=5000, num_colors=8, cache_path=dataset_path)
    model = StructuralShapeNet(8, 10)
    model.load_state_dict(torch.load(model_path))
    inference = ShapeInference(model)

    # Get sample
    sample = dataset.get_raw_sample(sample_idx)
    test_grid = sample['grid']
    h, w = test_grid.shape[:2]

    # Ground truth
    gt_vertex = sample['vertex_map']
    gt_vertices = np.argwhere(gt_vertex > 0.5)

    # Run inference
    prediction = inference.predict(test_grid)
    pred_vertex_map = prediction['raw_outputs']['vertex_map']

    # Try different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    print("="*80)
    print(f"SAMPLE {sample_idx} DIAGNOSIS")
    print("="*80)

    # Ground truth info
    if 'metadata' in sample:
        meta = sample['metadata']
        if 'type' in meta:
            print(f"\nGround truth shape: {meta['type']}")
        elif 'shapes' in meta:
            print(f"\nGround truth shapes: {[s['type'] for s in meta['shapes']]}")

    print(f"\nGround truth vertices: {len(gt_vertices)}")
    print(f"GT vertex locations: {gt_vertices.tolist()}")

    print(f"\nPredicted shape(s): {[obj['type'] for obj in prediction['objects']]}")
    print(f"Predicted object count: {len(prediction['objects'])}")

    # Analyze vertex detection at different thresholds
    print("\n" + "-"*80)
    print("VERTEX DETECTION AT DIFFERENT THRESHOLDS")
    print("-"*80)

    for thresh in thresholds:
        pred_vertices = np.argwhere(pred_vertex_map > thresh)
        print(f"\nThreshold {thresh}:")
        print(f"  Detected vertices: {len(pred_vertices)}")
        print(f"  Locations: {pred_vertices.tolist()[:10]}")  # First 10

        # Calculate accuracy
        if len(gt_vertices) > 0:
            correct = 0
            for gv in gt_vertices:
                if len(pred_vertices) > 0:
                    distances = [np.linalg.norm(gv - pv) for pv in pred_vertices]
                    if min(distances) < 1.5:
                        correct += 1
            accuracy = correct / len(gt_vertices)
            print(f"  Accuracy: {accuracy*100:.1f}%")

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Sample {sample_idx} Diagnosis', fontsize=14, fontweight='bold')

    input_rgb = test_grid.argmax(axis=2)

    # Input
    axes[0, 0].imshow(input_rgb, cmap='tab10', interpolation='nearest')
    add_gridlines(axes[0, 0], h, w)
    axes[0, 0].set_title('Input')
    axes[0, 0].axis('off')

    # GT vertices
    axes[0, 1].imshow(gt_vertex, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    add_gridlines(axes[0, 1], h, w)
    for vy, vx in gt_vertices:
        axes[0, 1].plot(vx, vy, 'g*', markersize=15, markeredgewidth=2, markeredgecolor='white')
    axes[0, 1].set_title(f'GT Vertices ({len(gt_vertices)})')
    axes[0, 1].axis('off')

    # Pred vertex heatmap
    axes[0, 2].imshow(pred_vertex_map, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    add_gridlines(axes[0, 2], h, w)
    axes[0, 2].set_title('Pred Vertex Heatmap')
    axes[0, 2].axis('off')

    # Pred vertices at different thresholds
    for i, thresh in enumerate([0.2, 0.3, 0.5]):
        ax = axes[1, i]
        ax.imshow(input_rgb, cmap='tab10', alpha=0.5, interpolation='nearest')
        add_gridlines(ax, h, w)

        pred_vertices = np.argwhere(pred_vertex_map > thresh)
        for vy, vx in pred_vertices:
            ax.plot(vx, vy, 'r*', markersize=12, markeredgewidth=1.5, markeredgecolor='white')

        # Show clustered vertices from actual prediction
        if thresh == 0.3:
            for obj in prediction['objects']:
                for vy, vx in obj['vertices']:
                    ax.plot(vx, vy, 'bo', markersize=8, fillstyle='none', markeredgewidth=2)

        ax.set_title(f'Threshold {thresh} ({len(pred_vertices)} verts)')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Print detailed info about predicted objects
    print("\n" + "-"*80)
    print("PREDICTED OBJECTS DETAIL")
    print("-"*80)
    for i, obj in enumerate(prediction['objects']):
        print(f"\nObject {i+1}:")
        print(f"  Type: {obj['type']}")
        print(f"  Detected vertices: {len(obj['vertices'])}")
        print(f"  Vertex locations: {obj['vertices']}")
        print(f"  Area: {obj['properties']['area']}")
        print(f"  Filled: {obj['properties']['is_filled']}")


if __name__ == "__main__":
    import sys

    dataset_path = 'data/train_dataset_large.pkl'
    model_path = 'best_model.pth'

    if len(sys.argv) > 1:
        sample_idx = int(sys.argv[1])
    else:
        # Use first failing rectangle example
        sample_idx = 1390

    diagnose_sample(dataset_path, model_path, sample_idx)

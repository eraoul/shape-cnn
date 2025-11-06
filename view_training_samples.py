"""
View Training Set Samples with Model Predictions
Shows how the model performs on actual training data
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from data_generation import ShapeDataset
from model import StructuralShapeNet
from inference import ShapeInference


def add_gridlines(ax, h, w, color='white', linewidth=0.5, alpha=0.3):
    """Add gridlines to visualize grid cells."""
    for x in np.arange(0.5, w, 1):
        ax.axvline(x, color=color, linewidth=linewidth, alpha=alpha)
    for y in np.arange(0.5, h, 1):
        ax.axhline(y, color=color, linewidth=linewidth, alpha=alpha)


class TrainingSetViewer:
    """Interactive viewer for training set samples."""

    def __init__(self, dataset_path, model_path='best_model.pth', num_colors=8, max_instances=10):
        # Load dataset
        print(f"Loading dataset from {dataset_path}...")
        # When loading from cache, num_samples is ignored and loaded from file
        self.dataset = ShapeDataset(num_samples=5000, num_colors=num_colors,
                                    cache_path=dataset_path)

        # Load model
        print("Loading model...")
        self.model = StructuralShapeNet(num_colors, max_instances)
        self.model.load_state_dict(torch.load(model_path))
        self.inference = ShapeInference(self.model)
        print(f"Using device: {self.inference.device}")
        print(f"Dataset size: {len(self.dataset)} samples")

    def visualize_sample(self, idx):
        """Visualize a specific sample from the training set."""
        if idx >= len(self.dataset):
            print(f"Index {idx} out of range. Dataset has {len(self.dataset)} samples.")
            return

        # Get raw sample (numpy arrays)
        sample = self.dataset.get_raw_sample(idx)
        test_grid = sample['grid']
        h, w = test_grid.shape[:2]

        # Get ground truth
        gt_instance = sample['instance_map']
        gt_vertex = sample['vertex_map']
        gt_edge = sample['edge_map']

        # Run inference
        prediction = self.inference.predict(test_grid)

        # Create figure with 3 rows: input, ground truth, predictions
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f'Training Sample {idx} | Grid: {h}x{w}',
                     fontsize=14, fontweight='bold')

        input_rgb = test_grid.argmax(axis=2)

        # Row 1: Input and ground truth
        axes[0, 0].imshow(input_rgb, cmap='tab10', interpolation='nearest')
        add_gridlines(axes[0, 0], h, w)
        axes[0, 0].set_title('Input Grid', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(gt_instance, cmap='tab10', interpolation='nearest')
        add_gridlines(axes[0, 1], h, w)
        axes[0, 1].set_title('GT Instances', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(gt_vertex, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        add_gridlines(axes[0, 2], h, w)
        axes[0, 2].set_title('GT Vertices', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')

        # Row 2: Predicted outputs
        axes[1, 0].imshow(prediction['raw_outputs']['instance_map'],
                         cmap='tab10', interpolation='nearest')
        add_gridlines(axes[1, 0], h, w)
        axes[1, 0].set_title('Pred Instances', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(prediction['raw_outputs']['vertex_map'],
                         cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        add_gridlines(axes[1, 1], h, w)
        axes[1, 1].set_title('Pred Vertices', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(prediction['raw_outputs']['edge_map'],
                         cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        add_gridlines(axes[1, 2], h, w)
        axes[1, 2].set_title('Pred Edges', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')

        # Row 3: Comparisons
        axes[2, 0].imshow(gt_edge, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        add_gridlines(axes[2, 0], h, w)
        axes[2, 0].set_title('GT Edges', fontsize=12, fontweight='bold')
        axes[2, 0].axis('off')

        # Overlay with detected vertices
        axes[2, 1].imshow(input_rgb, cmap='tab10', alpha=0.6, interpolation='nearest')
        add_gridlines(axes[2, 1], h, w)
        for obj in prediction['objects']:
            for vy, vx in obj['vertices']:
                axes[2, 1].plot(vx, vy, 'r*', markersize=15,
                              markeredgewidth=2, markeredgecolor='white')
        axes[2, 1].set_title('Detected Vertices', fontsize=12, fontweight='bold')
        axes[2, 1].axis('off')

        # Text summary
        axes[2, 2].axis('off')

        # Get ground truth info from metadata if available
        text = f"GROUND TRUTH:\n"
        if 'metadata' in sample:
            meta = sample['metadata']
            if 'type' in meta:
                text += f"  Type: {meta['type']}\n"
            if 'shapes' in meta:
                text += f"  Shapes: {len(meta['shapes'])}\n"
                for i, shape in enumerate(meta['shapes']):
                    text += f"    {i+1}. {shape['type']}\n"

        text += f"\nPREDICTED:\n"
        text += f"  Objects: {len(prediction['objects'])}\n\n"

        for i, obj in enumerate(prediction['objects']):
            text += f"Object {i+1}: {obj['type'].upper()}\n"
            text += f"  Color: {obj['color']}\n"
            text += f"  Vertices: {obj['properties']['num_vertices']}\n"
            text += f"  Area: {obj['properties']['area']} cells\n"
            text += f"  Filled: {obj['properties']['is_filled']}\n\n"

        axes[2, 2].text(0.05, 0.95, text, fontsize=10, verticalalignment='top',
                       fontfamily='monospace', transform=axes[2, 2].transAxes,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        axes[2, 2].set_title('Analysis', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.show()

    def view_samples(self, start_idx=0, count=None):
        """
        View samples one at a time from the training set.

        Args:
            start_idx: Starting index
            count: Number of samples to view. If None, continues until user closes window.
        """
        print("\n" + "="*60)
        print("TRAINING SET VIEWER")
        print("="*60)
        print("\nClose the window to view the next sample.")
        print("Press Ctrl+C in terminal to exit.\n")

        idx = start_idx
        max_idx = len(self.dataset)

        while count is None or idx < start_idx + count:
            if idx >= max_idx:
                print(f"\nReached end of dataset ({max_idx} samples).")
                break

            try:
                print(f"Viewing sample {idx}/{max_idx-1}...")
                self.visualize_sample(idx)
                idx += 1
            except KeyboardInterrupt:
                print("\n\nExiting viewer.")
                break


def view_random_training_samples(dataset_path, model_path='best_model.pth', count=5):
    """View random samples from training set."""
    import random

    viewer = TrainingSetViewer(dataset_path, model_path)

    print("\n" + "="*60)
    print(f"Viewing {count} random training samples")
    print("="*60)

    indices = random.sample(range(len(viewer.dataset)), min(count, len(viewer.dataset)))

    for i, idx in enumerate(indices):
        print(f"\nSample {i+1}/{count} (index {idx})...")
        viewer.visualize_sample(idx)


if __name__ == "__main__":
    import sys

    # Default paths
    dataset_path = 'data/train_dataset.pkl'
    model_path = 'best_model.pth'

    if len(sys.argv) > 1:
        if sys.argv[1] == '--random':
            # View random samples
            count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            view_random_training_samples(dataset_path, model_path, count)
        elif sys.argv[1] == '--val':
            # View validation set
            dataset_path = 'data/val_dataset.pkl'
            viewer = TrainingSetViewer(dataset_path, model_path)
            viewer.view_samples()
        else:
            # View starting from specific index
            start_idx = int(sys.argv[1])
            viewer = TrainingSetViewer(dataset_path, model_path)
            viewer.view_samples(start_idx=start_idx)
    else:
        # View training samples sequentially
        viewer = TrainingSetViewer(dataset_path, model_path)
        viewer.view_samples()

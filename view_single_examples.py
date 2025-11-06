"""
View Single Examples from the Shape Recognition Pipeline
Shows one example at a time with interactive navigation
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from data_generation_with_types import ShapeGeneratorWithTypes
from model_large import LargeShapeNet
from model_small import SmallShapeNet
from inference_with_classification import ShapeInferenceWithClassification
from skimage.draw import polygon as draw_polygon, line as draw_line


def add_gridlines(ax, h, w, color='white', linewidth=0.5, alpha=0.3):
    """Add gridlines to visualize grid cells."""
    # Vertical lines
    for x in np.arange(0.5, w, 1):
        ax.axvline(x, color=color, linewidth=linewidth, alpha=alpha)
    # Horizontal lines
    for y in np.arange(0.5, h, 1):
        ax.axhline(y, color=color, linewidth=linewidth, alpha=alpha)


class ExampleViewer:
    """Interactive viewer for shape recognition examples."""

    def __init__(self, model_path='best_model_small.pth', num_colors=10, max_instances=10):
        # Load model
        print("Loading model...")

        # Auto-detect model class based on filename
        if 'large' in model_path:
            print("Using LargeShapeNet")
            self.model = LargeShapeNet(num_colors, max_instances)
        else:
            print("Using SmallShapeNet (small model)")
            self.model = SmallShapeNet(num_colors, max_instances)

        self.model.load_state_dict(torch.load(model_path))
        self.inference = ShapeInferenceWithClassification(self.model)
        print(f"Using device: {self.inference.device}")

        # Setup generator
        self.generator = ShapeGeneratorWithTypes(num_colors, min_size=5, max_size=25)

        # Current sample
        self.current_sample = None
        self.current_prediction = None

    def generate_new_sample(self):
        """Generate a new random sample."""
        self.current_sample = self.generator.generate_sample()
        test_grid = self.current_sample['grid']
        self.current_prediction = self.inference.predict(test_grid)

    def reconstruct_grid(self, objects, h, w):
        """Reconstruct grid from detected objects for comparison."""
        reconstructed = np.zeros((h, w, 10))  # 10 colors

        for obj in objects:
            color = obj['color']
            vertices = obj['vertices']
            is_filled = obj['properties']['is_filled']

            if len(vertices) < 2:
                continue

            # Draw edges
            for i in range(len(vertices)):
                v1 = vertices[i]
                v2 = vertices[(i + 1) % len(vertices)]
                rr, cc = draw_line(int(v1[0]), int(v1[1]), int(v2[0]), int(v2[1]))
                # Clip to bounds
                valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                rr, cc = rr[valid], cc[valid]
                reconstructed[rr, cc, color] = 1.0

            # Fill if needed
            if is_filled and len(vertices) > 2:
                verts_array = np.array(vertices)
                rr, cc = draw_polygon(verts_array[:, 0], verts_array[:, 1], shape=(h, w))
                reconstructed[rr, cc, color] = 1.0

        return reconstructed

    def print_detection_details(self, prediction):
        """Print all detection details to console."""
        h, w = prediction['grid_size']

        print("\n" + "="*80)
        print(f"DETECTION RESULTS - Grid Size: {h}x{w}")
        print("="*80)

        # Shape classification statistics
        shape_class_map = prediction['raw_outputs']['shape_class_map']
        print("\nShape Classification Head Output:")
        print("-" * 80)
        unique_classes, counts = np.unique(shape_class_map, return_counts=True)
        for class_id, count in zip(unique_classes, counts):
            class_name = SmallShapeNet.get_shape_name(class_id)
            percentage = 100 * count / (h * w)
            print(f"  Class {class_id} ({class_name:12s}): {count:4d} pixels ({percentage:5.1f}%)")

        # Detected objects
        print(f"\nDetected Objects: {len(prediction['objects'])}")
        print("="*80)

        for i, obj in enumerate(prediction['objects']):
            print(f"\nObject {i+1}:")
            print(f"  Type:       {obj['type'].upper()}")
            print(f"  Color:      {obj['color']}")
            print(f"  Area:       {obj['properties']['area']} cells")
            print(f"  Filled:     {obj['properties']['is_filled']}")
            print(f"  Vertices:   {obj['properties']['num_vertices']}")

            # Vertex coordinates table
            if len(obj['vertices']) > 0:
                print(f"\n  Vertex Coordinates:")
                print(f"    {'Vertex':<8s} {'Y':>8s} {'X':>8s}")
                print(f"    {'-'*8} {'-'*8} {'-'*8}")
                for vi, (vy, vx) in enumerate(obj['vertices']):
                    print(f"    V{vi+1:<7d} {vy:8.2f} {vx:8.2f}")

            # Edge segments
            if len(obj['vertices']) >= 2:
                print(f"\n  Edge Segments:")
                print(f"    {'Edge':<8s} {'From (Y,X)':>18s} {'To (Y,X)':>18s} {'Length':>8s}")
                print(f"    {'-'*8} {'-'*18} {'-'*18} {'-'*8}")
                vertices = obj['vertices']
                for vi in range(len(vertices)):
                    v1 = vertices[vi]
                    v2 = vertices[(vi + 1) % len(vertices)]
                    length = np.sqrt((v2[0]-v1[0])**2 + (v2[1]-v1[1])**2)
                    print(f"    E{vi+1:<7d} ({v1[0]:6.2f}, {v1[1]:6.2f}) ({v2[0]:6.2f}, {v2[1]:6.2f}) {length:8.2f}")

            # Shape-specific info
            if 'dimensions' in obj:
                print(f"\n  Rectangle Dimensions:")
                print(f"    Width:  {obj['dimensions']['width']:.2f}")
                print(f"    Height: {obj['dimensions']['height']:.2f}")

            if 'angles' in obj:
                print(f"\n  Triangle Angles:")
                for ai, angle in enumerate(obj['angles']):
                    print(f"    Angle {ai+1}: {angle:.1f}°")

            if 'circle_params' in obj:
                print(f"\n  Circle Parameters:")
                print(f"    Center: ({obj['circle_params']['center'][0]:.2f}, {obj['circle_params']['center'][1]:.2f})")
                print(f"    Radius: {obj['circle_params']['radius']:.2f}")

        print("\n" + "="*80)

    def visualize_current(self):
        """Visualize the current sample."""
        if self.current_sample is None:
            self.generate_new_sample()

        test_grid = self.current_sample['grid']
        prediction = self.current_prediction
        h, w = test_grid.shape[:2]

        # Print all details to console
        self.print_detection_details(prediction)

        # Reconstruct grid from detections
        reconstructed_grid = self.reconstruct_grid(prediction['objects'], h, w)

        # Create figure with 2x3 layout
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Grid Size: {h}x{w} | Objects: {len(prediction["objects"])}',
                     fontsize=14, fontweight='bold')

        # Input grid
        input_rgb = test_grid.argmax(axis=2)
        axes[0, 0].imshow(input_rgb, cmap='tab10', interpolation='nearest', vmin=0, vmax=9)
        add_gridlines(axes[0, 0], h, w)
        axes[0, 0].set_title('Input Grid', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # Instance segmentation
        axes[0, 1].imshow(prediction['raw_outputs']['instance_map'],
                         cmap='tab10', interpolation='nearest')
        add_gridlines(axes[0, 1], h, w)
        axes[0, 1].set_title('Instance Segmentation', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        # Shape Classification Head Output (NEW)
        axes[0, 2].imshow(prediction['raw_outputs']['shape_class_map'],
                         cmap='viridis', interpolation='nearest', vmin=0, vmax=3)
        add_gridlines(axes[0, 2], h, w)
        axes[0, 2].set_title('Shape Classification\n(0=bg, 1=line, 2=rect, 3=cell)',
                            fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')

        # Vertex heatmap
        axes[1, 0].imshow(prediction['raw_outputs']['vertex_map'],
                         cmap='hot', interpolation='nearest')
        add_gridlines(axes[1, 0], h, w)
        axes[1, 0].set_title('Vertex Heatmap', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')

        # Edge heatmap
        axes[1, 1].imshow(prediction['raw_outputs']['edge_map'],
                         cmap='hot', interpolation='nearest')
        add_gridlines(axes[1, 1], h, w)
        axes[1, 1].set_title('Edge Heatmap', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')

        # Reconstructed Grid (NEW)
        reconstructed_rgb = reconstructed_grid.argmax(axis=2)
        axes[1, 2].imshow(reconstructed_rgb, cmap='tab10', interpolation='nearest', vmin=0, vmax=9)
        add_gridlines(axes[1, 2], h, w)
        axes[1, 2].set_title('Reconstructed from Detections', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

    def view_examples(self, count=None):
        """
        View examples one at a time.

        Args:
            count: Number of examples to view. If None, continues until user closes window.
        """
        print("\n" + "="*60)
        print("SHAPE RECOGNITION EXAMPLE VIEWER")
        print("="*60)
        print("\nClose the window to view the next example.")
        print("Press Ctrl+C in terminal to exit.\n")

        idx = 0
        while count is None or idx < count:
            try:
                print(f"Generating example {idx+1}...")
                self.generate_new_sample()
                self.visualize_current()
                idx += 1
            except KeyboardInterrupt:
                print("\n\nExiting viewer.")
                break


def view_single_example(model_path='best_model_small.pth'):
    """Quick function to view a single example."""
    viewer = ExampleViewer(model_path)
    viewer.generate_new_sample()
    viewer.visualize_current()


def view_multiple_examples(count=10, model_path='best_model_small.pth'):
    """View multiple examples one at a time."""
    viewer = ExampleViewer(model_path)
    viewer.view_examples(count)


if __name__ == "__main__":
    import sys

    model_path = 'best_model_small.pth'  # Default to small model
    count = None

    # Parse arguments
    if len(sys.argv) > 1:
        # First arg could be count or model path
        if sys.argv[1].endswith('.pth'):
            model_path = sys.argv[1]
        else:
            count = int(sys.argv[1])

    if len(sys.argv) > 2:
        # Second arg is model path
        if sys.argv[2].endswith('.pth'):
            model_path = sys.argv[2]

    # View examples
    if count is not None:
        view_multiple_examples(count, model_path)
    else:
        viewer = ExampleViewer(model_path)
        viewer.view_examples()

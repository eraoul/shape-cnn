"""
Visualize Keypoint Predictions
Shows predicted vs ground truth keypoints and reconstructed shapes
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from data_generation_with_keypoints import ShapeGeneratorWithKeypoints
from model_with_keypoints import ShapeNetWithKeypoints
from inference_with_keypoints import KeypointInference
from skimage.draw import line as draw_line, polygon as draw_polygon


def add_gridlines(ax, h, w, color='white', linewidth=0.5, alpha=0.3):
    """Add gridlines to visualize grid cells."""
    for x in np.arange(0.5, w, 1):
        ax.axvline(x, color=color, linewidth=linewidth, alpha=alpha)
    for y in np.arange(0.5, h, 1):
        ax.axhline(y, color=color, linewidth=linewidth, alpha=alpha)


def reconstruct_grid_from_keypoints(objects, h, w, num_colors=10):
    """Reconstruct grid from detected keypoints."""
    reconstructed = np.zeros((h, w, num_colors))

    for obj in objects:
        color = obj['color']
        vertices = obj['vertices']
        is_filled = obj['properties']['is_filled']

        if len(vertices) < 2:
            # Single cell
            if len(vertices) == 1:
                y, x = int(vertices[0][0]), int(vertices[0][1])
                if 0 <= y < h and 0 <= x < w:
                    reconstructed[y, x, color] = 1.0
            continue

        # Draw edges
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]
            rr, cc = draw_line(int(v1[0]), int(v1[1]), int(v2[0]), int(v2[1]))
            valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            rr, cc = rr[valid], cc[valid]
            reconstructed[rr, cc, color] = 1.0

        # Fill if needed
        if is_filled and len(vertices) > 2:
            verts_array = np.array(vertices)
            try:
                rr, cc = draw_polygon(verts_array[:, 0], verts_array[:, 1], shape=(h, w))
                reconstructed[rr, cc, color] = 1.0
            except:
                pass  # Skip if polygon is invalid

    return reconstructed


class KeypointExampleViewer:
    """Interactive viewer for keypoint predictions."""

    def __init__(self, model_path='best_model_keypoints.pth', num_colors=10, max_instances=10):
        # Load model
        print("Loading model...")
        self.model = ShapeNetWithKeypoints(num_colors, max_instances)
        self.model.load_state_dict(torch.load(model_path))
        self.inference = KeypointInference(self.model)
        print(f"Using device: {self.inference.device}")

        # Setup generator
        self.generator = ShapeGeneratorWithKeypoints(num_colors, min_size=5, max_size=25, max_instances=max_instances)

        # Current sample
        self.current_sample = None
        self.current_prediction = None

    def generate_new_sample(self):
        """Generate a new random sample."""
        self.current_sample = self.generator.generate_sample()
        self.current_prediction = self.inference.predict(self.current_sample['grid'])

    def visualize_current(self):
        """Visualize the current sample."""
        if self.current_sample is None:
            self.generate_new_sample()

        grid = self.current_sample['grid']
        gt_keypoints = self.current_sample['keypoint_targets']
        prediction = self.current_prediction
        h, w = self.current_sample['grid_size']

        # Print details to console
        self.print_detection_details(prediction, gt_keypoints, h, w)

        # Reconstruct grid from keypoints
        reconstructed_grid = reconstruct_grid_from_keypoints(prediction['objects'], h, w)

        # Create figure with 2x3 layout
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Grid Size: {h}x{w} | Objects: {len(prediction["objects"])}',
                     fontsize=14, fontweight='bold')

        # Input grid
        input_rgb = grid.argmax(axis=2)
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

        # Shape Classification
        axes[0, 2].imshow(prediction['raw_outputs']['shape_class_map'],
                         cmap='viridis', interpolation='nearest', vmin=0, vmax=3)
        add_gridlines(axes[0, 2], h, w)
        axes[0, 2].set_title('Shape Classification\n(0=bg, 1=line, 2=rect, 3=cell)',
                            fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')

        # Ground Truth Keypoints
        axes[1, 0].imshow(input_rgb, cmap='tab10', interpolation='nearest', vmin=0, vmax=9, alpha=0.5)
        axes[1, 0].set_title('Ground Truth Keypoints', fontsize=12, fontweight='bold')

        # Plot GT keypoints
        for inst_idx in range(len(gt_keypoints)):
            kp_array = gt_keypoints[inst_idx]
            for kp_idx in range(len(kp_array)):
                y_norm, x_norm, valid = kp_array[kp_idx]
                if valid > 0.5:
                    y, x = y_norm * h, x_norm * w
                    axes[1, 0].plot(x, y, 'go', markersize=15, markerfacecolor='lime',
                                  markeredgewidth=2, markeredgecolor='darkgreen')

        add_gridlines(axes[1, 0], h, w)
        axes[1, 0].set_xlim(-0.5, w - 0.5)
        axes[1, 0].set_ylim(h - 0.5, -0.5)
        axes[1, 0].axis('off')

        # Predicted Keypoints
        axes[1, 1].imshow(input_rgb, cmap='tab10', interpolation='nearest', vmin=0, vmax=9, alpha=0.5)
        axes[1, 1].set_title('Predicted Keypoints', fontsize=12, fontweight='bold')

        # Plot predicted keypoints
        for obj in prediction['objects']:
            for vy, vx in obj['vertices']:
                axes[1, 1].plot(vx, vy, 'c^', markersize=15, markerfacecolor='cyan',
                              markeredgewidth=2, markeredgecolor='blue')

        add_gridlines(axes[1, 1], h, w)
        axes[1, 1].set_xlim(-0.5, w - 0.5)
        axes[1, 1].set_ylim(h - 0.5, -0.5)
        axes[1, 1].axis('off')

        # Reconstructed Grid
        reconstructed_rgb = reconstructed_grid.argmax(axis=2)
        axes[1, 2].imshow(reconstructed_rgb, cmap='tab10', interpolation='nearest', vmin=0, vmax=9)
        add_gridlines(axes[1, 2], h, w)
        axes[1, 2].set_title('Reconstructed from Keypoints', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

    def print_detection_details(self, prediction, gt_keypoints, h, w):
        """Print all detection details to console."""
        print("\n" + "="*80)
        print(f"KEYPOINT DETECTION RESULTS - Grid Size: {h}x{w}")
        print("="*80)

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

            # Vertex coordinates
            if len(obj['vertices']) > 0:
                print(f"\n  Predicted Keypoints:")
                print(f"    {'Vertex':<8s} {'Y':>8s} {'X':>8s}")
                print(f"    {'-'*8} {'-'*8} {'-'*8}")
                for vi, (vy, vx) in enumerate(obj['vertices']):
                    print(f"    V{vi+1:<7d} {vy:8.2f} {vx:8.2f}")

                # Compare with ground truth if available
                inst_idx = obj['id'] - 1
                if inst_idx < len(gt_keypoints):
                    gt_kp = gt_keypoints[inst_idx]
                    print(f"\n  Ground Truth Keypoints:")
                    print(f"    {'Vertex':<8s} {'Y':>8s} {'X':>8s} {'Error':>8s}")
                    print(f"    {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

                    for kp_idx in range(len(gt_kp)):
                        y_norm, x_norm, valid = gt_kp[kp_idx]
                        if valid > 0.5:
                            gt_y, gt_x = y_norm * h, x_norm * w

                            # Find closest predicted keypoint
                            if kp_idx < len(obj['vertices']):
                                pred_y, pred_x = obj['vertices'][kp_idx]
                                error = np.sqrt((gt_y - pred_y)**2 + (gt_x - pred_x)**2)
                                print(f"    V{kp_idx+1:<7d} {gt_y:8.2f} {gt_x:8.2f} {error:8.2f}")
                            else:
                                print(f"    V{kp_idx+1:<7d} {gt_y:8.2f} {gt_x:8.2f} {'MISSING':>8s}")

            # Shape-specific info
            if 'dimensions' in obj:
                print(f"\n  Rectangle Dimensions:")
                print(f"    Width:  {obj['dimensions']['width']:.2f}")
                print(f"    Height: {obj['dimensions']['height']:.2f}")

            if 'length' in obj:
                print(f"\n  Line Length: {obj['length']:.2f}")

        print("\n" + "="*80)

    def view_examples(self, count=None):
        """
        View examples one at a time.

        Args:
            count: Number of examples to view. If None, continues until user closes window.
        """
        print("\n" + "="*60)
        print("KEYPOINT PREDICTION EXAMPLE VIEWER")
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


if __name__ == "__main__":
    import sys

    model_path = 'best_model_keypoints.pth'
    count = None

    # Parse arguments
    if len(sys.argv) > 1:
        if sys.argv[1].endswith('.pth'):
            model_path = sys.argv[1]
        else:
            count = int(sys.argv[1])

    if len(sys.argv) > 2:
        if sys.argv[2].endswith('.pth'):
            model_path = sys.argv[2]

    # View examples
    viewer = KeypointExampleViewer(model_path)
    viewer.view_examples(count)

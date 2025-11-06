"""
Inference script for U-Net shape detection.

Loads trained model and detects shapes in grid images.
Returns list of shapes with vertices and classifications.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
from skimage.measure import label, regionprops

from model_unet import CompactShapeUNet
from data_generation_improved import SHAPE_CLASSES


# Reverse mapping
SHAPE_ID_TO_NAME = {v: k for k, v in SHAPE_CLASSES.items()}


class ShapeDetector:
    """Inference wrapper for shape detection."""

    def __init__(
        self,
        model_path: str,
        num_colors: int = 10,
        max_instances: int = 10,
        num_shape_classes: int = 6,
        device: str = None,
        vertex_threshold: float = 0.5,
        edge_threshold: float = 0.5
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            num_colors: Number of color channels
            max_instances: Maximum number of instances model can detect
            num_shape_classes: Number of shape classes
            device: Device to run inference on (None=auto-detect MPS/CUDA/CPU)
            vertex_threshold: Threshold for vertex detection (0-1)
            edge_threshold: Threshold for edge detection (0-1)
        """
        # Auto-detect device if not specified: MPS (Apple Metal) > CUDA (NVIDIA) > CPU
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = device
        self.num_colors = num_colors
        self.max_instances = max_instances
        self.vertex_threshold = vertex_threshold
        self.edge_threshold = edge_threshold

        # Load model
        print(f"Loading model from {model_path}...")
        self.model = CompactShapeUNet(
            num_colors=num_colors,
            max_instances=max_instances,
            num_shape_classes=num_shape_classes,
            bilinear=True
        ).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("Model loaded successfully!")

    def predict(self, grid: np.ndarray) -> Dict:
        """
        Detect shapes in a grid.

        Args:
            grid: (H, W, num_colors) one-hot encoded grid

        Returns:
            dict with:
            - shapes: List of detected shapes with vertices and classification
            - instance_map: (H, W) instance segmentation
            - vertex_map: (H, W) vertex heatmap
            - edge_map: (H, W) edge heatmap
        """
        h, w = grid.shape[:2]

        # Convert to tensor
        grid_tensor = torch.from_numpy(grid).float().permute(2, 0, 1).unsqueeze(0)
        grid_tensor = grid_tensor.to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(grid_tensor)

        # Extract predictions
        instance_pred = outputs['instances'][0].cpu().numpy()  # (max_instances+1, H, W)
        instance_map = instance_pred.argmax(axis=0)  # (H, W)

        vertex_pred = outputs['vertices'][0, 0].cpu().numpy()  # (H, W)
        vertex_map = (vertex_pred > self.vertex_threshold).astype(np.float32)

        edge_pred = outputs['edges'][0, 0].cpu().numpy()  # (H, W)
        edge_map = (edge_pred > self.edge_threshold).astype(np.float32)

        shape_class_pred = outputs['shape_classes'][0].cpu().numpy()  # (num_classes, H, W)
        shape_class_map = shape_class_pred.argmax(axis=0)  # (H, W)

        # Extract shapes from instance map
        shapes = self._extract_shapes(
            grid, instance_map, vertex_map, edge_map, shape_class_map
        )

        return {
            'shapes': shapes,
            'instance_map': instance_map,
            'vertex_map': vertex_map,
            'edge_map': edge_map,
            'shape_class_map': shape_class_map,
        }

    def _extract_shapes(
        self,
        grid: np.ndarray,
        instance_map: np.ndarray,
        vertex_map: np.ndarray,
        edge_map: np.ndarray,
        shape_class_map: np.ndarray
    ) -> List[Dict]:
        """Extract individual shape information from predictions."""
        shapes = []
        h, w = instance_map.shape

        # Get unique instance IDs (excluding background 0)
        instance_ids = np.unique(instance_map)
        instance_ids = instance_ids[instance_ids > 0]

        for inst_id in instance_ids:
            # Get mask for this instance
            mask = (instance_map == inst_id)

            if mask.sum() < 2:
                continue  # Skip tiny instances

            # Get color
            color = self._get_dominant_color(grid, mask)

            # Get shape type (most common non-background class in mask)
            shape_classes_in_mask = shape_class_map[mask]
            shape_classes_in_mask = shape_classes_in_mask[shape_classes_in_mask > 0]
            if len(shape_classes_in_mask) > 0:
                shape_class_id = np.bincount(shape_classes_in_mask).argmax()
                shape_type = SHAPE_ID_TO_NAME.get(shape_class_id, 'unknown')
            else:
                shape_type = 'unknown'

            # Extract vertices for this instance
            vertices_in_mask = vertex_map & mask
            vertex_coords = np.argwhere(vertices_in_mask)  # (N, 2) in (y, x) format

            # If no vertices detected, use region properties
            if len(vertex_coords) < 2:
                vertex_coords = self._extract_vertices_from_mask(mask, shape_type)

            # Order vertices
            if len(vertex_coords) >= 2:
                vertex_coords = self._order_vertices(vertex_coords, shape_type)

            # Get all pixels in shape
            pixels = np.argwhere(mask).tolist()

            # Get bounding box
            props = regionprops(mask.astype(int))[0]
            bbox = props.bbox  # (min_row, min_col, max_row, max_col)

            shapes.append({
                'instance_id': int(inst_id),
                'type': shape_type,
                'color': int(color),
                'vertices': vertex_coords.tolist(),
                'num_vertices': len(vertex_coords),
                'pixels': pixels,
                'bbox': bbox,
                'area': mask.sum(),
            })

        return shapes

    def _get_dominant_color(self, grid: np.ndarray, mask: np.ndarray) -> int:
        """Get the dominant non-background color in a mask."""
        # Sum over masked region
        masked_grid = grid * mask[:, :, np.newaxis]
        color_sums = masked_grid.sum(axis=(0, 1))

        # Exclude background (color 0)
        color_sums[0] = -1
        return color_sums.argmax()

    def _extract_vertices_from_mask(
        self,
        mask: np.ndarray,
        shape_type: str
    ) -> np.ndarray:
        """Extract vertices from shape mask using contour analysis."""
        # Get contour
        from skimage.measure import find_contours

        contours = find_contours(mask.astype(float), 0.5)
        if len(contours) == 0:
            # Fallback: just use corners of bounding box
            ys, xs = np.where(mask)
            return np.array([
                [ys.min(), xs.min()],
                [ys.min(), xs.max()],
                [ys.max(), xs.max()],
                [ys.max(), xs.min()],
            ])

        contour = contours[0]  # Take largest contour

        # Simplify contour to find corners
        # Use Douglas-Peucker algorithm via approximate_polygon
        from skimage.measure import approximate_polygon

        if shape_type == 'line':
            # For lines, just take endpoints
            if len(contour) >= 2:
                # Find two most distant points
                from scipy.spatial.distance import pdist, squareform
                dists = squareform(pdist(contour))
                i, j = np.unravel_index(dists.argmax(), dists.shape)
                vertices = contour[[i, j]]
            else:
                vertices = contour
        elif shape_type == 'triangle':
            # Approximate to 3 vertices
            tolerance = 2.0
            vertices = approximate_polygon(contour, tolerance=tolerance)
            while len(vertices) > 4 and tolerance < 10:
                tolerance += 0.5
                vertices = approximate_polygon(contour, tolerance=tolerance)
        elif shape_type == 'rectangle':
            # Approximate to 4 vertices
            tolerance = 2.0
            vertices = approximate_polygon(contour, tolerance=tolerance)
            while len(vertices) > 5 and tolerance < 10:
                tolerance += 0.5
                vertices = approximate_polygon(contour, tolerance=tolerance)
        else:
            # General polygon: adaptive simplification
            tolerance = 2.0
            vertices = approximate_polygon(contour, tolerance=tolerance)

        # Convert to (y, x) integer coordinates
        vertices = np.round(vertices).astype(int)

        return vertices

    def _order_vertices(
        self,
        vertices: np.ndarray,
        shape_type: str
    ) -> np.ndarray:
        """Order vertices consistently (e.g., counterclockwise)."""
        if len(vertices) < 2:
            return vertices

        if shape_type == 'line' and len(vertices) >= 2:
            # For lines, just keep two endpoints
            if len(vertices) > 2:
                # Find two most distant points
                from scipy.spatial.distance import pdist, squareform
                dists = squareform(pdist(vertices))
                i, j = np.unravel_index(dists.argmax(), dists.shape)
                return vertices[[i, j]]
            return vertices

        # For polygons, order by angle from centroid
        centroid = vertices.mean(axis=0)
        angles = np.arctan2(
            vertices[:, 0] - centroid[0],
            vertices[:, 1] - centroid[1]
        )
        sorted_indices = np.argsort(angles)
        return vertices[sorted_indices]


def visualize_prediction(
    grid: np.ndarray,
    prediction: Dict,
    save_path: Optional[str] = None
):
    """Visualize detection results."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import matplotlib.patches as mpatches

    h, w = grid.shape[:2]

    # Convert one-hot grid to RGB for visualization
    rgb = np.zeros((h, w, 3))
    for y in range(h):
        for x in range(w):
            color_idx = np.argmax(grid[y, x])
            if color_idx == 0:
                rgb[y, x] = [1, 1, 1]  # Background white
            else:
                # Consistent color per index
                np.random.seed(color_idx)
                rgb[y, x] = np.random.rand(3)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original grid
    axes[0, 0].imshow(rgb, interpolation='nearest')
    axes[0, 0].set_title('Original Grid')
    axes[0, 0].axis('off')

    # Instance segmentation
    axes[0, 1].imshow(prediction['instance_map'], cmap='tab20', interpolation='nearest')
    axes[0, 1].set_title('Instance Segmentation')
    axes[0, 1].axis('off')

    # Vertex map
    axes[0, 2].imshow(prediction['vertex_map'], cmap='hot', interpolation='nearest')
    axes[0, 2].set_title('Vertex Detection')
    axes[0, 2].axis('off')

    # Edge map
    axes[1, 0].imshow(prediction['edge_map'], cmap='hot', interpolation='nearest')
    axes[1, 0].set_title('Edge Detection')
    axes[1, 0].axis('off')

    # Shape classification
    axes[1, 1].imshow(prediction['shape_class_map'], cmap='tab10', interpolation='nearest')
    axes[1, 1].set_title('Shape Classification')
    axes[1, 1].axis('off')

    # Detected shapes with vertices
    axes[1, 2].imshow(rgb, interpolation='nearest')
    for shape in prediction['shapes']:
        vertices = np.array(shape['vertices'])
        if len(vertices) >= 2:
            # Draw vertices
            axes[1, 2].scatter(vertices[:, 1], vertices[:, 0], c='red', s=50, marker='x')

            # Draw polygon outline
            if len(vertices) >= 3:
                # Close the polygon
                poly_vertices = np.vstack([vertices, vertices[0]])
                axes[1, 2].plot(poly_vertices[:, 1], poly_vertices[:, 0], 'r-', linewidth=2)
            elif len(vertices) == 2:
                # Draw line
                axes[1, 2].plot(vertices[:, 1], vertices[:, 0], 'r-', linewidth=2)

            # Add label
            centroid = vertices.mean(axis=0)
            axes[1, 2].text(
                centroid[1], centroid[0],
                f"{shape['type']}\n({shape['num_vertices']}v)",
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                fontsize=8
            )

    axes[1, 2].set_title(f"Detected Shapes ({len(prediction['shapes'])})")
    axes[1, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def print_detection_results(prediction: Dict):
    """Print detection results in readable format."""
    print("\n" + "="*60)
    print(f"DETECTED SHAPES: {len(prediction['shapes'])}")
    print("="*60)

    for i, shape in enumerate(prediction['shapes'], 1):
        print(f"\nShape {i}:")
        print(f"  Type: {shape['type']}")
        print(f"  Color: {shape['color']}")
        print(f"  Vertices ({shape['num_vertices']}):")
        for j, (y, x) in enumerate(shape['vertices']):
            print(f"    {j+1}. ({y}, {x})")
        print(f"  Area: {shape['area']} pixels")
        print(f"  Bounding Box: {shape['bbox']}")

    print("\n" + "="*60)


if __name__ == '__main__':
    import sys

    # Test inference on generated samples
    from data_generation_improved import generate_dataset

    print("Generating test samples...")
    test_samples = generate_dataset(
        num_samples=10,
        num_colors=10,
        min_size=10,
        max_size=25,
        cache_path=None
    )

    # Create detector
    model_path = 'checkpoints/unet_best.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        print("Please train the model first using train_unet.py")
        sys.exit(1)

    detector = ShapeDetector(
        model_path=model_path,
        vertex_threshold=0.4,
        edge_threshold=0.4
    )

    # Run inference on test samples
    import os
    os.makedirs('results', exist_ok=True)

    for i, sample in enumerate(test_samples[:5]):
        print(f"\n{'='*60}")
        print(f"Processing sample {i+1}/5...")
        print(f"{'='*60}")

        # Ground truth info
        print(f"\nGround Truth:")
        for meta in sample['metadata']:
            print(f"  - {meta['type']} (color {meta['color']}, {len(meta['vertices'])} vertices)")

        # Run inference
        prediction = detector.predict(sample['grid'])

        # Print results
        print_detection_results(prediction)

        # Visualize
        visualize_prediction(
            sample['grid'],
            prediction,
            save_path=f'results/detection_{i+1}.png'
        )

    print("\n" + "="*60)
    print("Inference complete! Results saved to results/")
    print("="*60)

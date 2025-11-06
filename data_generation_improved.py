"""
Improved data generation for shape detection.

Generates diverse training data with:
- Lines (2 vertices)
- Triangles (3 vertices)
- Rectangles (4 vertices, axis-aligned and rotated)
- Arbitrary polygons (5-8 vertices)
- Circles (approximated as regular polygons)

Key improvements:
- All shape types represented
- Better vertex/edge annotations
- Configurable difficulty levels
- Data augmentation support
"""

import numpy as np
from skimage.draw import line as draw_line, polygon as draw_polygon
import pickle
from typing import Dict, List, Tuple, Optional
import math


SHAPE_CLASSES = {
    'background': 0,
    'line': 1,
    'triangle': 2,
    'rectangle': 3,
    'polygon': 4,  # 5-8 sided
    'circle': 5,
}

NUM_SHAPE_CLASSES = len(SHAPE_CLASSES)


class ImprovedShapeGenerator:
    """Generate training samples with diverse shape types."""

    def __init__(
        self,
        num_colors: int = 10,
        min_size: int = 5,
        max_size: int = 30,
        max_instances: int = 10,
        shape_distribution: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            num_colors: Number of unique colors (including background)
            min_size: Minimum grid dimension
            max_size: Maximum grid dimension
            max_instances: Maximum number of shape instances per sample
            shape_distribution: Probability distribution over shape types
        """
        self.num_colors = num_colors
        self.min_size = min_size
        self.max_size = max_size
        self.max_instances = max_instances

        # Default: uniform distribution over all shape types
        if shape_distribution is None:
            shape_types = ['line', 'triangle', 'rectangle', 'polygon', 'circle']
            shape_distribution = {s: 1.0/len(shape_types) for s in shape_types}

        self.shape_types = list(shape_distribution.keys())
        self.shape_probs = [shape_distribution[s] for s in self.shape_types]

    def generate_sample(self) -> Dict:
        """Generate a single training sample."""
        # Random grid size
        h = np.random.randint(self.min_size, self.max_size + 1)
        w = np.random.randint(self.min_size, self.max_size + 1)

        # Initialize outputs
        grid = np.zeros((h, w, self.num_colors), dtype=np.float32)
        grid[:, :, 0] = 1.0  # Background color

        instance_map = np.zeros((h, w), dtype=np.int32)
        vertex_map = np.zeros((h, w), dtype=np.float32)
        edge_map = np.zeros((h, w), dtype=np.float32)
        shape_type_map = np.zeros((h, w), dtype=np.int32)

        # Determine number of shapes based on grid size
        area = h * w
        if area < 25:
            num_shapes = 1
        elif area < 100:
            num_shapes = np.random.randint(1, 3)
        else:
            max_shapes = min(self.max_instances, max(1, area // 50))
            num_shapes = np.random.randint(1, max_shapes + 1)

        # Track used colors
        used_colors = {0}  # Background
        metadata = []

        # Generate each shape
        for instance_id in range(1, num_shapes + 1):
            # Pick random color
            available_colors = list(set(range(1, self.num_colors)) - used_colors)
            if not available_colors:
                break
            color = np.random.choice(available_colors)
            used_colors.add(color)

            # Pick shape type
            shape_type = np.random.choice(self.shape_types, p=self.shape_probs)

            # Generate shape vertices
            vertices = self._generate_shape_vertices(shape_type, h, w)

            if vertices is None or len(vertices) < 2:
                continue

            # Rasterize shape
            filled = shape_type != 'line' and np.random.rand() > 0.3
            shape_class_id = SHAPE_CLASSES[shape_type]

            success = self._rasterize_shape(
                grid, instance_map, vertex_map, edge_map, shape_type_map,
                vertices, color, instance_id, shape_class_id, filled
            )

            if success:
                metadata.append({
                    'instance_id': instance_id,
                    'type': shape_type,
                    'color': color,
                    'filled': filled,
                    'vertices': vertices,
                })

        return {
            'grid': grid,
            'instance_map': instance_map,
            'vertex_map': vertex_map,
            'edge_map': edge_map,
            'shape_type_map': shape_type_map,
            'metadata': metadata,
        }

    def _generate_shape_vertices(
        self,
        shape_type: str,
        h: int,
        w: int
    ) -> Optional[List[Tuple[int, int]]]:
        """Generate vertices for a given shape type."""

        if shape_type == 'line':
            return self._generate_line_vertices(h, w)
        elif shape_type == 'triangle':
            return self._generate_triangle_vertices(h, w)
        elif shape_type == 'rectangle':
            return self._generate_rectangle_vertices(h, w)
        elif shape_type == 'polygon':
            return self._generate_polygon_vertices(h, w)
        elif shape_type == 'circle':
            return self._generate_circle_vertices(h, w)
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")

    def _generate_line_vertices(self, h: int, w: int) -> List[Tuple[int, int]]:
        """Generate 2 random points for a line."""
        # Ensure line is reasonably long (at least 2 pixels)
        min_length = 2
        max_attempts = 20

        for _ in range(max_attempts):
            y1, x1 = np.random.randint(0, h), np.random.randint(0, w)
            y2, x2 = np.random.randint(0, h), np.random.randint(0, w)

            length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            if length >= min_length:
                return [(y1, x1), (y2, x2)]

        # Fallback: horizontal or vertical line
        if w >= h:
            y = np.random.randint(0, h)
            return [(y, 0), (y, w - 1)]
        else:
            x = np.random.randint(0, w)
            return [(0, x), (h - 1, x)]

    def _generate_triangle_vertices(self, h: int, w: int) -> Optional[List[Tuple[int, int]]]:
        """Generate 3 points for a triangle."""
        min_area = 2.0
        max_attempts = 30

        for _ in range(max_attempts):
            vertices = []
            for _ in range(3):
                y = np.random.randint(0, h)
                x = np.random.randint(0, w)
                vertices.append((y, x))

            # Check if triangle has sufficient area (not degenerate)
            area = self._polygon_area(vertices)
            if area >= min_area:
                return vertices

        # Fallback: create a simple triangle
        cy, cx = h // 2, w // 2
        size = min(h, w) // 3
        return [
            (max(0, cy - size), cx),
            (min(h - 1, cy + size), max(0, cx - size)),
            (min(h - 1, cy + size), min(w - 1, cx + size)),
        ]

    def _generate_rectangle_vertices(self, h: int, w: int) -> Optional[List[Tuple[int, int]]]:
        """Generate 4 points for a rectangle (possibly rotated)."""
        min_side = 2

        # 50% chance of axis-aligned rectangle
        if np.random.rand() < 0.5:
            # Axis-aligned
            y1 = np.random.randint(0, max(1, h - min_side))
            x1 = np.random.randint(0, max(1, w - min_side))
            y2 = np.random.randint(y1 + min_side, h)
            x2 = np.random.randint(x1 + min_side, w)

            return [
                (y1, x1),
                (y1, x2),
                (y2, x2),
                (y2, x1),
            ]
        else:
            # Rotated rectangle
            cy = np.random.randint(min_side, h - min_side)
            cx = np.random.randint(min_side, w - min_side)

            # Random dimensions
            rect_h = np.random.randint(min_side, min(h // 2, cy, h - cy) + 1)
            rect_w = np.random.randint(min_side, min(w // 2, cx, w - cx) + 1)

            # Random rotation
            angle = np.random.uniform(0, np.pi)

            # Four corners before rotation
            corners = [
                (-rect_h / 2, -rect_w / 2),
                (-rect_h / 2, rect_w / 2),
                (rect_h / 2, rect_w / 2),
                (rect_h / 2, -rect_w / 2),
            ]

            # Rotate and translate
            vertices = []
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            for dy, dx in corners:
                rot_y = dy * cos_a - dx * sin_a
                rot_x = dy * sin_a + dx * cos_a
                y = int(round(cy + rot_y))
                x = int(round(cx + rot_x))
                y = np.clip(y, 0, h - 1)
                x = np.clip(x, 0, w - 1)
                vertices.append((y, x))

            return vertices

    def _generate_polygon_vertices(self, h: int, w: int) -> Optional[List[Tuple[int, int]]]:
        """Generate 5-8 vertices for an arbitrary polygon."""
        num_vertices = np.random.randint(5, 9)
        min_area = 3.0
        max_attempts = 30

        for _ in range(max_attempts):
            # Generate random vertices
            vertices = []
            for _ in range(num_vertices):
                y = np.random.randint(0, h)
                x = np.random.randint(0, w)
                vertices.append((y, x))

            # Order by angle from centroid (create convex-ish polygon)
            vertices = self._order_vertices_by_angle(vertices)

            # Check area
            area = self._polygon_area(vertices)
            if area >= min_area:
                return vertices

        # Fallback: create regular polygon
        return self._generate_regular_polygon(h, w, num_vertices)

    def _generate_circle_vertices(self, h: int, w: int) -> List[Tuple[int, int]]:
        """Generate vertices approximating a circle (regular polygon with 12-20 sides)."""
        num_vertices = np.random.randint(12, 21)
        return self._generate_regular_polygon(h, w, num_vertices)

    def _generate_regular_polygon(self, h: int, w: int, num_vertices: int) -> List[Tuple[int, int]]:
        """Generate a regular polygon centered in grid."""
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 3

        vertices = []
        for i in range(num_vertices):
            angle = 2 * np.pi * i / num_vertices
            y = int(round(cy + radius * np.sin(angle)))
            x = int(round(cx + radius * np.cos(angle)))
            y = np.clip(y, 0, h - 1)
            x = np.clip(x, 0, w - 1)
            vertices.append((y, x))

        return vertices

    def _polygon_area(self, vertices: List[Tuple[int, int]]) -> float:
        """Calculate polygon area using shoelace formula."""
        if len(vertices) < 3:
            return 0.0

        area = 0.0
        for i in range(len(vertices)):
            y1, x1 = vertices[i]
            y2, x2 = vertices[(i + 1) % len(vertices)]
            area += x1 * y2 - x2 * y1

        return abs(area) / 2.0

    def _order_vertices_by_angle(self, vertices: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Order vertices by angle from centroid (counterclockwise)."""
        if len(vertices) < 3:
            return vertices

        # Calculate centroid
        cy = np.mean([v[0] for v in vertices])
        cx = np.mean([v[1] for v in vertices])

        # Sort by angle
        def angle(v):
            return np.arctan2(v[0] - cy, v[1] - cx)

        return sorted(vertices, key=angle)

    def _rasterize_shape(
        self,
        grid: np.ndarray,
        instance_map: np.ndarray,
        vertex_map: np.ndarray,
        edge_map: np.ndarray,
        shape_type_map: np.ndarray,
        vertices: List[Tuple[int, int]],
        color: int,
        instance_id: int,
        shape_class_id: int,
        filled: bool
    ) -> bool:
        """Rasterize a shape onto the grid."""
        h, w = grid.shape[:2]

        # Mark vertices
        for vy, vx in vertices:
            if 0 <= vy < h and 0 <= vx < w:
                vertex_map[vy, vx] = 1.0

        # Draw edges
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]

            try:
                rr, cc = draw_line(int(v1[0]), int(v1[1]), int(v2[0]), int(v2[1]))

                # Filter to valid coordinates
                valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                rr, cc = rr[valid], cc[valid]

                if len(rr) > 0:
                    edge_map[rr, cc] = 1.0
                    grid[rr, cc, 0] = 0.0  # Remove background
                    grid[rr, cc, color] = 1.0
                    instance_map[rr, cc] = instance_id
                    shape_type_map[rr, cc] = shape_class_id
            except Exception as e:
                print(f"Warning: Failed to draw edge: {e}")
                continue

        # Fill interior if needed
        if filled and len(vertices) >= 3:
            try:
                verts_array = np.array(vertices)
                rr, cc = draw_polygon(verts_array[:, 0], verts_array[:, 1], shape=(h, w))

                if len(rr) > 0:
                    grid[rr, cc, 0] = 0.0  # Remove background
                    grid[rr, cc, color] = 1.0
                    instance_map[rr, cc] = instance_id
                    shape_type_map[rr, cc] = shape_class_id
            except Exception as e:
                print(f"Warning: Failed to fill polygon: {e}")

        return True


def generate_dataset(
    num_samples: int,
    num_colors: int = 10,
    min_size: int = 5,
    max_size: int = 30,
    cache_path: Optional[str] = None,
    **generator_kwargs
) -> List[Dict]:
    """Generate a dataset of training samples."""

    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"Generating {num_samples} samples...")
    generator = ImprovedShapeGenerator(
        num_colors=num_colors,
        min_size=min_size,
        max_size=max_size,
        **generator_kwargs
    )

    dataset = []
    for i in range(num_samples):
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{num_samples}")
        sample = generator.generate_sample()
        dataset.append(sample)

    if cache_path:
        print(f"Caching dataset to {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)

    return dataset


if __name__ == '__main__':
    import os

    # Generate a small test dataset
    print("Generating test dataset...")
    dataset = generate_dataset(
        num_samples=100,
        num_colors=10,
        min_size=5,
        max_size=30,
        cache_path='data/test_dataset_improved.pkl'
    )

    # Print statistics
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(dataset)}")

    shape_counts = {name: 0 for name in SHAPE_CLASSES.keys()}
    for sample in dataset:
        for meta in sample['metadata']:
            shape_counts[meta['type']] += 1

    print(f"  Shape distribution:")
    for shape, count in shape_counts.items():
        if count > 0:
            print(f"    {shape}: {count}")

    # Visualize a few samples
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, sample in enumerate(dataset[:6]):
        # Convert one-hot grid to RGB
        grid = sample['grid']
        h, w = grid.shape[:2]

        rgb = np.zeros((h, w, 3))
        for y in range(h):
            for x in range(w):
                color_idx = np.argmax(grid[y, x])
                if color_idx == 0:
                    rgb[y, x] = [0, 0, 0]  # Background black
                else:
                    # Random color per shape
                    np.random.seed(color_idx)
                    rgb[y, x] = np.random.rand(3)

        axes[i].imshow(rgb, interpolation='nearest')
        axes[i].set_title(f"Sample {i + 1}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('data/sample_shapes_improved.png', dpi=150)
    print("\nSaved visualization to data/sample_shapes_improved.png")

"""
Data Generation with Shape Type Labels
Extends data generation to include per-pixel shape classification labels
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.draw import polygon as draw_polygon, line as draw_line, disk as draw_disk
import pickle
import os

# Import shape class mappings from model
from model_small import ShapeNetWithClassification


class ShapeGeneratorWithTypes:
    """Generates synthetic training data with shape type labels."""

    def __init__(self, num_colors=8, min_size=1, max_size=30):
        self.num_colors = num_colors
        self.min_size = min_size
        self.max_size = max_size
        self.shape_classes = ShapeNetWithClassification.SHAPE_CLASSES

    def generate_sample(self):
        """Generate a single training sample with random grid size."""
        h = np.random.randint(self.min_size, self.max_size + 1)
        w = np.random.randint(self.min_size, self.max_size + 1)

        # Handle special cases
        if h == 1 and w == 1:
            return self._generate_single_cell(h, w)
        elif h == 1 or w == 1:
            return self._generate_line_grid(h, w)
        elif h <= 3 or w <= 3:
            return self._generate_small_grid(h, w)
        else:
            return self._generate_normal_grid(h, w)

    def _generate_single_cell(self, h, w):
        """Generate 1x1 grid."""
        color = np.random.randint(1, self.num_colors)  # Reserve 0 for background
        grid = np.zeros((h, w, self.num_colors))
        grid[0, 0, color] = 1

        shape_type_map = np.zeros((h, w), dtype=np.int32)
        shape_type_map[0, 0] = self.shape_classes['single_cell']

        return {
            'grid': grid,
            'instance_map': np.ones((h, w), dtype=np.int32),  # Instance 1
            'vertex_map': np.zeros((h, w), dtype=np.float32),
            'edge_map': np.zeros((h, w), dtype=np.float32),
            'shape_type_map': shape_type_map,
            'metadata': {'type': 'single_cell', 'color': color}
        }

    def _generate_line_grid(self, h, w):
        """Generate 1xW or Hx1 grid (line only)."""
        color = np.random.randint(1, self.num_colors)  # Reserve 0 for background
        grid = np.zeros((h, w, self.num_colors))
        grid[:, :, color] = 1

        instance_map = np.ones((h, w), dtype=np.int32)
        vertex_map = np.zeros((h, w), dtype=np.float32)
        edge_map = np.ones((h, w), dtype=np.float32)

        shape_type_map = np.ones((h, w), dtype=np.int32) * self.shape_classes['line']

        # Mark endpoints as vertices
        if h == 1:
            vertex_map[0, 0] = 1
            vertex_map[0, w-1] = 1
        else:
            vertex_map[0, 0] = 1
            vertex_map[h-1, 0] = 1

        return {
            'grid': grid,
            'instance_map': instance_map,
            'vertex_map': vertex_map,
            'edge_map': edge_map,
            'shape_type_map': shape_type_map,
            'metadata': {'type': 'line', 'color': color}
        }

    def _generate_small_grid(self, h, w):
        """Generate small grid with simple shapes (line or rectangle only)."""
        grid = np.zeros((h, w, self.num_colors))
        instance_map = np.zeros((h, w), dtype=np.int32)
        vertex_map = np.zeros((h, w), dtype=np.float32)
        edge_map = np.zeros((h, w), dtype=np.float32)
        shape_type_map = np.zeros((h, w), dtype=np.int32)  # background by default

        # Just one simple shape
        color = np.random.randint(1, self.num_colors)  # Reserve 0 for background
        shape_type = np.random.choice(['line', 'rectangle'])

        if shape_type == 'line':
            vertices = self._random_line(h, w)
        else:
            vertices = self._random_rectangle(h, w)

        self._rasterize_shape(grid, instance_map, vertex_map, edge_map, shape_type_map,
                             vertices, color, 1, shape_type, filled=True)

        return {
            'grid': grid,
            'instance_map': instance_map,
            'vertex_map': vertex_map,
            'edge_map': edge_map,
            'shape_type_map': shape_type_map,
            'metadata': {'type': shape_type, 'color': color}
        }

    def _generate_normal_grid(self, h, w):
        """Generate normal grid with multiple shapes (line and rectangle only)."""
        grid = np.zeros((h, w, self.num_colors))
        instance_map = np.zeros((h, w), dtype=np.int32)
        vertex_map = np.zeros((h, w), dtype=np.float32)
        edge_map = np.zeros((h, w), dtype=np.float32)
        shape_type_map = np.zeros((h, w), dtype=np.int32)  # background by default

        # Track which colors are used at each location to avoid conflicts
        color_used = np.zeros((h, w, self.num_colors), dtype=bool)

        area = h * w
        max_shapes = max(1, min(5, area // 25))
        num_shapes = np.random.randint(1, max_shapes + 1)

        shapes_metadata = []

        for shape_id in range(1, num_shapes + 1):
            # Only lines and rectangles now
            shape_type = np.random.choice(['line', 'rectangle'])
            filled = np.random.choice([True, False], p=[0.7, 0.3])

            if shape_type == 'line':
                vertices = self._random_line(h, w)
            else:
                vertices = self._random_rectangle(h, w)

            # Determine which pixels this shape will occupy
            shape_pixels = self._get_shape_pixels(vertices, h, w, filled)

            # Find an available color (not used in any overlapping pixels)
            color = self._find_available_color(shape_pixels, color_used)

            # Rasterize the shape
            self._rasterize_shape(grid, instance_map, vertex_map, edge_map, shape_type_map,
                                 vertices, color, shape_id, shape_type, filled)

            # Mark color as used in these pixels
            for (y, x) in shape_pixels:
                color_used[y, x, color] = True

            shapes_metadata.append({
                'type': shape_type,
                'color': color,
                'filled': filled,
                'vertices': vertices
            })

        return {
            'grid': grid,
            'instance_map': instance_map,
            'vertex_map': vertex_map,
            'edge_map': edge_map,
            'shape_type_map': shape_type_map,
            'metadata': {'shapes': shapes_metadata}
        }

    def _get_shape_pixels(self, vertices, h, w, filled):
        """Get all pixels that will be occupied by a shape."""
        pixels = set()

        # Add edge pixels
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]
            rr, cc = draw_line(int(v1[0]), int(v1[1]), int(v2[0]), int(v2[1]))
            valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            rr, cc = rr[valid], cc[valid]
            for y, x in zip(rr, cc):
                pixels.add((y, x))

        # Add filled pixels if needed
        if filled and len(vertices) > 2:
            verts_array = np.array(vertices)
            rr, cc = draw_polygon(verts_array[:, 0], verts_array[:, 1], shape=(h, w))
            for y, x in zip(rr, cc):
                pixels.add((y, x))

        return list(pixels)

    def _find_available_color(self, pixels, color_used):
        """Find a color that's not used in any of the given pixels."""
        # Try each color starting from 1 (0 is background)
        for color in range(1, self.num_colors):
            color_available = True
            for (y, x) in pixels:
                if color_used[y, x, color]:
                    color_available = False
                    break
            if color_available:
                return color

        # If all colors are used, just pick a random one from 1 to num_colors-1
        return np.random.randint(1, self.num_colors)

    def _random_line(self, h, w):
        """Generate random line vertices."""
        y1, x1 = np.random.randint(0, h), np.random.randint(0, w)
        y2, x2 = np.random.randint(0, h), np.random.randint(0, w)
        return [(y1, x1), (y2, x2)]

    def _random_triangle(self, h, w):
        """Generate random triangle vertices."""
        if h <= 2 or w <= 2:
            if h == 2 and w == 2:
                return [(0, 0), (0, 1), (1, 0)]
            elif h == 2:
                return [(0, 0), (0, w-1), (1, w//2)]
            else:
                return [(0, 0), (h-1, 0), (h//2, 1)]

        margin = 0
        vertices = []
        for _ in range(3):
            y = np.random.randint(margin, max(margin+1, h))
            x = np.random.randint(margin, max(margin+1, w))
            vertices.append((y, x))
        return vertices

    def _random_rectangle(self, h, w):
        """Generate random rectangle vertices."""
        if h <= 2 or w <= 2:
            return [(0, 0), (0, w-1), (h-1, w-1), (h-1, 0)]

        margin = 0
        y1 = np.random.randint(margin, max(margin+1, h-2))
        x1 = np.random.randint(margin, max(margin+1, w-2))

        max_height = max(1, min(5, h-y1-1))
        max_width = max(1, min(5, w-x1-1))

        height = np.random.randint(1, max(2, max_height))
        width = np.random.randint(1, max(2, max_width))

        return [
            (y1, x1),
            (y1, x1 + width),
            (y1 + height, x1 + width),
            (y1 + height, x1)
        ]

    def _random_circle(self, h, w):
        """Generate random circle (approximated as polygon)."""
        if h <= 4 or w <= 4:
            cy, cx = h // 2, w // 2
            radius = min(h, w) // 3 if min(h, w) >= 3 else 1
        else:
            margin = 2
            cy = np.random.randint(margin, max(margin+1, h-margin))
            cx = np.random.randint(margin, max(margin+1, w-margin))
            max_radius = max(1, min(h, w) // 3)
            radius = np.random.randint(1, max(2, max_radius))

        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        vertices = [
            (cy + radius * np.sin(a), cx + radius * np.cos(a))
            for a in angles
        ]
        return vertices

    def _random_irregular(self, h, w):
        """Generate random irregular polygon."""
        margin = 1
        num_vertices = np.random.randint(4, 7)
        vertices = []
        for _ in range(num_vertices):
            y = np.random.randint(margin, max(margin+1, h-margin))
            x = np.random.randint(margin, max(margin+1, w-margin))
            vertices.append((y, x))
        return vertices

    def _rasterize_shape(self, grid, instance_map, vertex_map, edge_map, shape_type_map,
                        vertices, color, instance_id, shape_type, filled=True):
        """Rasterize a shape into the grid with shape type labels."""
        h, w = grid.shape[:2]
        shape_class_id = self.shape_classes.get(shape_type, 0)

        # Mark vertices
        for vy, vx in vertices:
            vy, vx = int(np.clip(vy, 0, h-1)), int(np.clip(vx, 0, w-1))
            vertex_map[vy, vx] = 1.0

        # Draw edges
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]
            rr, cc = draw_line(int(v1[0]), int(v1[1]), int(v2[0]), int(v2[1]))
            # Clip to bounds
            valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            rr, cc = rr[valid], cc[valid]
            edge_map[rr, cc] = 1.0
            grid[rr, cc, color] = 1.0
            instance_map[rr, cc] = instance_id
            shape_type_map[rr, cc] = shape_class_id  # NEW: Mark shape type

        # Fill if needed
        if filled and len(vertices) > 2:
            verts_array = np.array(vertices)
            rr, cc = draw_polygon(verts_array[:, 0], verts_array[:, 1], shape=(h, w))
            grid[rr, cc, color] = 1.0
            instance_map[rr, cc] = instance_id
            shape_type_map[rr, cc] = shape_class_id  # NEW: Mark shape type


class ShapeDatasetWithTypes(Dataset):
    """PyTorch dataset for shape recognition with type labels."""

    def __init__(self, num_samples=1000, num_colors=8, min_size=1, max_size=30, cache_path=None):
        self.num_samples = num_samples
        self.num_colors = num_colors
        self.min_size = min_size
        self.max_size = max_size
        self.cache_path = cache_path

        # Try to load from cache
        if cache_path and os.path.exists(cache_path):
            print(f"Loading dataset from {cache_path}...")
            self.load(cache_path)
        else:
            # Generate new samples
            print(f"Generating {num_samples} new samples with type labels...")
            self.generator = ShapeGeneratorWithTypes(num_colors, min_size, max_size)
            self.samples = [self.generator.generate_sample() for _ in range(num_samples)]

            # Save to cache if path provided
            if cache_path:
                self.save(cache_path)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Convert to tensors and transpose to (C, H, W)
        grid = torch.from_numpy(sample['grid']).float().permute(2, 0, 1)
        instance_map = torch.from_numpy(sample['instance_map']).long()
        vertex_map = torch.from_numpy(sample['vertex_map']).float()
        edge_map = torch.from_numpy(sample['edge_map']).float()
        shape_type_map = torch.from_numpy(sample['shape_type_map']).long()

        return {
            'grid': grid,
            'instance_map': instance_map,
            'vertex_map': vertex_map,
            'edge_map': edge_map,
            'shape_type_map': shape_type_map
        }

    def save(self, path):
        """Save dataset to disk."""
        print(f"Saving dataset to {path}...")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        data = {
            'samples': self.samples,
            'num_samples': self.num_samples,
            'num_colors': self.num_colors,
            'min_size': self.min_size,
            'max_size': self.max_size
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Dataset saved successfully.")

    def load(self, path):
        """Load dataset from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.samples = data['samples']
        self.num_samples = data['num_samples']
        self.num_colors = data['num_colors']
        self.min_size = data['min_size']
        self.max_size = data['max_size']
        print(f"Dataset loaded: {self.num_samples} samples.")

    def get_raw_sample(self, idx):
        """Get raw sample without tensor conversion (useful for visualization)."""
        return self.samples[idx]


# Aliases for drop-in replacement
ShapeGenerator = ShapeGeneratorWithTypes
ShapeDataset = ShapeDatasetWithTypes

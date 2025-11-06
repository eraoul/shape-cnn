"""
Data Generation with Keypoint Targets
Generates data for keypoint regression training
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.draw import polygon as draw_polygon, line as draw_line
import pickle
import os

# Import shape class mappings from model
from model_with_keypoints import ShapeNetWithKeypoints


class ShapeGeneratorWithKeypoints:
    """Generates synthetic training data with keypoint targets."""

    def __init__(self, num_colors=10, min_size=1, max_size=30, max_instances=10):
        self.num_colors = num_colors
        self.min_size = min_size
        self.max_size = max_size
        self.max_instances = max_instances
        self.shape_classes = ShapeNetWithKeypoints.SHAPE_CLASSES
        self.max_keypoints = ShapeNetWithKeypoints.MAX_KEYPOINTS

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
        color = np.random.randint(1, self.num_colors)
        grid = np.zeros((h, w, self.num_colors))
        grid[0, 0, color] = 1

        shape_type_map = np.zeros((h, w), dtype=np.int32)
        shape_type_map[0, 0] = self.shape_classes['single_cell']

        # Keypoint targets: [max_instances, max_keypoints, 3]
        keypoint_targets = np.zeros((self.max_instances, self.max_keypoints, 3), dtype=np.float32)
        # Instance 1: single cell at (0, 0)
        keypoint_targets[0, 0, 0] = 0.0 / h  # y normalized
        keypoint_targets[0, 0, 1] = 0.0 / w  # x normalized
        keypoint_targets[0, 0, 2] = 1.0      # valid

        return {
            'grid': grid,
            'instance_map': np.ones((h, w), dtype=np.int32),
            'shape_type_map': shape_type_map,
            'keypoint_targets': keypoint_targets,
            'grid_size': (h, w),
            'metadata': {'type': 'single_cell', 'color': color}
        }

    def _generate_line_grid(self, h, w):
        """Generate 1xW or Hx1 grid (line only)."""
        color = np.random.randint(1, self.num_colors)
        grid = np.zeros((h, w, self.num_colors))
        grid[:, :, color] = 1

        instance_map = np.ones((h, w), dtype=np.int32)
        shape_type_map = np.ones((h, w), dtype=np.int32) * self.shape_classes['line']

        # Keypoint targets for line
        keypoint_targets = np.zeros((self.max_instances, self.max_keypoints, 3), dtype=np.float32)

        # Mark endpoints
        if h == 1:
            # Horizontal line
            keypoint_targets[0, 0, :] = [0.0 / h, 0.0 / w, 1.0]      # Start
            keypoint_targets[0, 1, :] = [0.0 / h, (w-1) / w, 1.0]   # End
        else:
            # Vertical line
            keypoint_targets[0, 0, :] = [0.0 / h, 0.0 / w, 1.0]      # Start
            keypoint_targets[0, 1, :] = [(h-1) / h, 0.0 / w, 1.0]   # End

        return {
            'grid': grid,
            'instance_map': instance_map,
            'shape_type_map': shape_type_map,
            'keypoint_targets': keypoint_targets,
            'grid_size': (h, w),
            'metadata': {'type': 'line', 'color': color}
        }

    def _generate_small_grid(self, h, w):
        """Generate small grid with simple shapes (line or rectangle only)."""
        grid = np.zeros((h, w, self.num_colors))
        instance_map = np.zeros((h, w), dtype=np.int32)
        shape_type_map = np.zeros((h, w), dtype=np.int32)

        # Just one simple shape
        color = np.random.randint(1, self.num_colors)
        shape_type = np.random.choice(['line', 'rectangle'])

        if shape_type == 'line':
            vertices = self._random_line(h, w)
        else:
            vertices = self._random_rectangle(h, w)

        self._rasterize_shape(grid, instance_map, shape_type_map,
                             vertices, color, 1, shape_type, filled=True)

        # Create keypoint targets
        keypoint_targets = self._create_keypoint_targets([(vertices, shape_type)], h, w)

        return {
            'grid': grid,
            'instance_map': instance_map,
            'shape_type_map': shape_type_map,
            'keypoint_targets': keypoint_targets,
            'grid_size': (h, w),
            'metadata': {'type': shape_type, 'color': color, 'vertices': vertices}
        }

    def _generate_normal_grid(self, h, w):
        """Generate normal grid with multiple shapes (line and rectangle only)."""
        grid = np.zeros((h, w, self.num_colors))
        instance_map = np.zeros((h, w), dtype=np.int32)
        shape_type_map = np.zeros((h, w), dtype=np.int32)

        # Track which colors are used at each location to avoid conflicts
        color_used = np.zeros((h, w, self.num_colors), dtype=bool)

        area = h * w
        max_shapes = max(1, min(self.max_instances, min(5, area // 25)))
        num_shapes = np.random.randint(1, max_shapes + 1)

        shapes_metadata = []
        shape_vertices_and_types = []

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
            self._rasterize_shape(grid, instance_map, shape_type_map,
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

            shape_vertices_and_types.append((vertices, shape_type))

        # Create keypoint targets
        keypoint_targets = self._create_keypoint_targets(shape_vertices_and_types, h, w)

        return {
            'grid': grid,
            'instance_map': instance_map,
            'shape_type_map': shape_type_map,
            'keypoint_targets': keypoint_targets,
            'grid_size': (h, w),
            'metadata': {'shapes': shapes_metadata}
        }

    def _create_keypoint_targets(self, shape_vertices_and_types, h, w):
        """
        Create keypoint targets tensor for all instances.

        Args:
            shape_vertices_and_types: List of (vertices, shape_type) tuples
            h, w: Grid dimensions

        Returns:
            keypoint_targets: [max_instances, max_keypoints, 3]
        """
        keypoint_targets = np.zeros((self.max_instances, self.max_keypoints, 3), dtype=np.float32)

        for inst_idx, (vertices, shape_type) in enumerate(shape_vertices_and_types):
            if inst_idx >= self.max_instances:
                break

            # Number of keypoints for this shape type
            num_kp = ShapeNetWithKeypoints.get_num_keypoints(shape_type)

            for kp_idx in range(min(num_kp, len(vertices), self.max_keypoints)):
                vy, vx = vertices[kp_idx]

                # Normalize coordinates to [0, 1]
                y_norm = np.clip(vy / h, 0, 1)
                x_norm = np.clip(vx / w, 0, 1)

                keypoint_targets[inst_idx, kp_idx, 0] = y_norm  # y
                keypoint_targets[inst_idx, kp_idx, 1] = x_norm  # x
                keypoint_targets[inst_idx, kp_idx, 2] = 1.0     # valid

        return keypoint_targets

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
        for color in range(1, self.num_colors):
            color_available = True
            for (y, x) in pixels:
                if color_used[y, x, color]:
                    color_available = False
                    break
            if color_available:
                return color

        return np.random.randint(1, self.num_colors)

    def _random_line(self, h, w):
        """Generate random line vertices."""
        y1, x1 = np.random.randint(0, h), np.random.randint(0, w)
        y2, x2 = np.random.randint(0, h), np.random.randint(0, w)
        return [(y1, x1), (y2, x2)]

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

    def _rasterize_shape(self, grid, instance_map, shape_type_map,
                        vertices, color, instance_id, shape_type, filled=True):
        """Rasterize a shape into the grid with shape type labels."""
        h, w = grid.shape[:2]
        shape_class_id = self.shape_classes.get(shape_type, 0)

        # Draw edges
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]
            rr, cc = draw_line(int(v1[0]), int(v1[1]), int(v2[0]), int(v2[1]))
            # Clip to bounds
            valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            rr, cc = rr[valid], cc[valid]
            grid[rr, cc, color] = 1.0
            instance_map[rr, cc] = instance_id
            shape_type_map[rr, cc] = shape_class_id

        # Fill if needed
        if filled and len(vertices) > 2:
            verts_array = np.array(vertices)
            rr, cc = draw_polygon(verts_array[:, 0], verts_array[:, 1], shape=(h, w))
            grid[rr, cc, color] = 1.0
            instance_map[rr, cc] = instance_id
            shape_type_map[rr, cc] = shape_class_id


class ShapeDatasetWithKeypoints(Dataset):
    """PyTorch dataset for shape recognition with keypoint targets."""

    def __init__(self, num_samples=1000, num_colors=10, min_size=1, max_size=30,
                 max_instances=10, cache_path=None):
        self.num_samples = num_samples
        self.num_colors = num_colors
        self.min_size = min_size
        self.max_size = max_size
        self.max_instances = max_instances
        self.cache_path = cache_path

        # Try to load from cache
        if cache_path and os.path.exists(cache_path):
            print(f"Loading dataset from {cache_path}...")
            self.load(cache_path)
        else:
            # Generate new samples
            print(f"Generating {num_samples} new samples with keypoint targets...")
            self.generator = ShapeGeneratorWithKeypoints(num_colors, min_size, max_size, max_instances)
            self.samples = [self.generator.generate_sample() for _ in range(num_samples)]

            # Save to cache if path provided
            if cache_path:
                self.save(cache_path)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Convert to tensors
        grid = torch.from_numpy(sample['grid']).float().permute(2, 0, 1)  # [C, H, W]
        instance_map = torch.from_numpy(sample['instance_map']).long()   # [H, W]
        shape_type_map = torch.from_numpy(sample['shape_type_map']).long()  # [H, W]
        keypoint_targets = torch.from_numpy(sample['keypoint_targets']).float()  # [max_instances, max_keypoints, 3]

        return {
            'grid': grid,
            'instance_map': instance_map,
            'shape_type_map': shape_type_map,
            'keypoint_targets': keypoint_targets,
            'original_sizes': sample['grid_size']
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
            'max_size': self.max_size,
            'max_instances': self.max_instances
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
        self.max_instances = data.get('max_instances', 10)
        print(f"Dataset loaded: {self.num_samples} samples.")

    def get_raw_sample(self, idx):
        """Get raw sample without tensor conversion (useful for visualization)."""
        return self.samples[idx]

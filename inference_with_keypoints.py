"""
Inference Pipeline with Keypoint Regression
Uses direct keypoint predictions from the neural network
"""

import numpy as np
import torch
from model_with_keypoints import ShapeNetWithKeypoints


class KeypointInference:
    """Inference pipeline using keypoint regression."""

    def __init__(self, model, device=None):
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def predict(self, grid):
        """
        Predict shapes from a grid using keypoint regression.

        Args:
            grid: numpy array of shape (H, W, num_colors) with one-hot encoded colors

        Returns:
            Dictionary with structured shape information
        """
        h, w, c = grid.shape

        # Convert to tensor
        grid_tensor = torch.from_numpy(grid).float().permute(2, 0, 1).unsqueeze(0)
        grid_tensor = grid_tensor.to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(grid_tensor)

        # Convert to numpy
        instance_map = torch.argmax(outputs['instances'][0], dim=0).cpu().numpy()[:h, :w]
        shape_class_map = torch.argmax(outputs['shape_classes'][0], dim=0).cpu().numpy()[:h, :w]
        keypoints = outputs['keypoints'][0].cpu().numpy()  # [max_instances, max_keypoints, 3]

        # Extract structured information
        objects = self._extract_objects(grid, instance_map, shape_class_map, keypoints, h, w)

        return {
            'grid_size': (h, w),
            'objects': objects,
            'raw_outputs': {
                'instance_map': instance_map,
                'shape_class_map': shape_class_map,
                'keypoints': keypoints
            }
        }

    def _extract_objects(self, grid, instance_map, shape_class_map, keypoints, h, w):
        """Extract structured object information using keypoint predictions."""
        objects = []

        # Find unique instances (excluding background 0)
        instance_ids = np.unique(instance_map)
        instance_ids = instance_ids[instance_ids > 0]

        for inst_id in instance_ids:
            mask = (instance_map == inst_id)

            # Get cells belonging to this instance
            cells = np.argwhere(mask).tolist()

            # Extract color
            color = self._extract_color(grid, mask)

            # Get shape type from neural network predictions
            shape_type = self._get_predicted_shape_type(mask, shape_class_map)

            # Extract keypoints for this instance (using 0-based indexing)
            inst_idx = inst_id - 1  # Instance IDs are 1-based, array is 0-based
            if inst_idx < len(keypoints):
                vertices = self._extract_keypoints(keypoints[inst_idx], shape_type, h, w)
            else:
                vertices = []

            # Compute properties
            area = len(cells)
            is_filled = self._is_filled_from_shape(shape_type, mask)

            obj = {
                'id': int(inst_id),
                'type': shape_type,
                'color': int(color),
                'cells': cells,
                'vertices': vertices,
                'properties': {
                    'area': area,
                    'is_filled': is_filled,
                    'num_vertices': len(vertices)
                }
            }

            # Add shape-specific parameters
            if shape_type == 'rectangle' and len(vertices) >= 4:
                obj['dimensions'] = self._compute_rectangle_dimensions(vertices[:4])
            elif shape_type == 'line' and len(vertices) >= 2:
                obj['length'] = self._compute_line_length(vertices[:2])

            objects.append(obj)

        return objects

    def _extract_keypoints(self, kp_array, shape_type, h, w):
        """
        Extract valid keypoints and denormalize coordinates.

        Args:
            kp_array: [max_keypoints, 3] array with (y_norm, x_norm, validity)
            shape_type: Shape type string
            h, w: Grid dimensions

        Returns:
            List of (y, x) tuples for valid keypoints
        """
        vertices = []
        num_expected = ShapeNetWithKeypoints.get_num_keypoints(shape_type)

        for i in range(min(num_expected, len(kp_array))):
            y_norm, x_norm, validity = kp_array[i]

            # Check if keypoint is valid
            if validity > 0.5:
                # Denormalize coordinates and ROUND to nearest integer grid cell
                y = np.round(y_norm * h)
                x = np.round(x_norm * w)

                # Clip to grid bounds
                y = np.clip(y, 0, h - 1)
                x = np.clip(x, 0, w - 1)

                vertices.append((int(y), int(x)))

        return vertices

    def _get_predicted_shape_type(self, mask, shape_class_map):
        """
        Get shape type from neural network predictions.
        Uses majority voting within the instance mask.
        """
        # Get predicted class IDs within this mask
        predicted_classes = shape_class_map[mask]

        # Remove background (class 0)
        predicted_classes = predicted_classes[predicted_classes > 0]

        if len(predicted_classes) == 0:
            return 'unknown'

        # Majority vote
        unique, counts = np.unique(predicted_classes, return_counts=True)
        majority_class_id = unique[np.argmax(counts)]

        # Convert class ID to name
        shape_name = ShapeNetWithKeypoints.get_shape_name(majority_class_id)

        return shape_name

    def _extract_color(self, grid, mask):
        """Extract dominant color from masked region."""
        masked_grid = grid[mask]
        if len(masked_grid) == 0:
            return 0
        color_sums = masked_grid.sum(axis=0)
        return np.argmax(color_sums)

    def _is_filled_from_shape(self, shape_type, mask):
        """Determine if shape is filled based on shape type and mask properties."""
        if mask.sum() < 4:
            return False

        # For rectangles and polygons, check if interior is filled
        # Simple heuristic: if most pixels are not on the perimeter, it's filled
        if shape_type in ['rectangle', 'single_cell']:
            # If area is small, assume filled
            if mask.sum() < 10:
                return True

            # Check density: filled shapes have high density
            y_coords, x_coords = np.where(mask)
            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()

            bounding_area = (y_max - y_min + 1) * (x_max - x_min + 1)
            density = mask.sum() / (bounding_area + 1e-6)

            return density > 0.5

        # Lines are never filled
        return False

    def _compute_rectangle_dimensions(self, vertices):
        """Compute width and height of rectangle."""
        if len(vertices) < 4:
            return {}

        v = np.array(vertices[:4])
        # Compute edge lengths
        edges = [np.linalg.norm(v[i] - v[(i+1) % 4]) for i in range(4)]
        edges.sort()

        return {
            'width': round(edges[0], 2),
            'height': round(edges[2], 2)
        }

    def _compute_line_length(self, vertices):
        """Compute length of line."""
        if len(vertices) < 2:
            return 0.0

        v1, v2 = np.array(vertices[0]), np.array(vertices[1])
        return float(np.linalg.norm(v2 - v1))

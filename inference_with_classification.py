"""
Inference Pipeline with Learned Shape Classification
Uses neural network predictions for shape types instead of heuristics
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.measure import find_contours, approximate_polygon
from model_small import ShapeNetWithClassification


class ShapeInferenceWithClassification:
    """Inference pipeline using learned shape classification."""

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
        Predict shapes from a grid using learned classification.

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
        vertex_map = outputs['vertices'][0, 0].cpu().numpy()[:h, :w]
        edge_map = outputs['edges'][0, 0].cpu().numpy()[:h, :w]
        shape_class_map = torch.argmax(outputs['shape_classes'][0], dim=0).cpu().numpy()[:h, :w]

        # Extract structured information
        objects = self._extract_objects(grid, instance_map, vertex_map, edge_map, shape_class_map)

        return {
            'grid_size': (h, w),
            'objects': objects,
            'raw_outputs': {
                'instance_map': instance_map,
                'vertex_map': vertex_map,
                'edge_map': edge_map,
                'shape_class_map': shape_class_map
            }
        }

    def _extract_objects(self, grid, instance_map, vertex_map, edge_map, shape_class_map):
        """Extract structured object information using learned shape types."""
        h, w = instance_map.shape
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

            # Find vertices
            vertices = self._find_vertices(mask, vertex_map)

            # LEARNED: Get shape type from neural network predictions
            shape_type = self._get_predicted_shape_type(mask, shape_class_map)

            # Order vertices based on shape type
            vertices = self._order_vertices(vertices, shape_type, mask)

            # Trace perimeter
            perimeter = self._trace_perimeter(mask)

            # Determine if filled
            is_filled = self._is_filled(mask, edge_map)

            # Compute properties
            area = len(cells)

            obj = {
                'id': int(inst_id),
                'type': shape_type,
                'color': int(color),
                'cells': cells,
                'vertices': vertices,
                'perimeter': perimeter,
                'properties': {
                    'area': area,
                    'is_filled': is_filled,
                    'num_vertices': len(vertices)
                }
            }

            # Add shape-specific parameters
            if shape_type == 'triangle' and len(vertices) >= 3:
                obj['angles'] = self._compute_triangle_angles(vertices[:3])
            elif shape_type == 'rectangle' and len(vertices) >= 4:
                obj['dimensions'] = self._compute_rectangle_dimensions(vertices[:4])
            elif shape_type == 'circle' and len(vertices) > 0:
                obj['circle_params'] = self._fit_circle(vertices)

            objects.append(obj)

        return objects

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
        shape_name = ShapeNetWithClassification.get_shape_name(majority_class_id)

        return shape_name

    def _extract_color(self, grid, mask):
        """Extract dominant color from masked region."""
        masked_grid = grid[mask]
        if len(masked_grid) == 0:
            return 0
        color_sums = masked_grid.sum(axis=0)
        return np.argmax(color_sums)

    def _find_vertices(self, mask, vertex_map, threshold=0.3):
        """Find vertex locations from mask and vertex heatmap."""
        vertices_in_mask = mask * (vertex_map > threshold)
        vertex_coords = np.argwhere(vertices_in_mask)

        # Cluster nearby vertices
        if len(vertex_coords) == 0:
            return []

        # Simple clustering: group vertices within distance 2
        clustered = []
        used = set()

        for i, coord in enumerate(vertex_coords):
            if i in used:
                continue

            cluster = [coord]
            for j, other_coord in enumerate(vertex_coords):
                if j <= i or j in used:
                    continue
                if np.linalg.norm(coord - other_coord) < 2.5:
                    cluster.append(other_coord)
                    used.add(j)

            # Use centroid of cluster
            centroid = np.mean(cluster, axis=0)
            clustered.append(tuple(centroid.tolist()))

        return clustered

    def _order_vertices(self, vertices, shape_type, mask):
        """
        Order vertices based on shape type for consistent reconstruction.

        Args:
            vertices: List of (y, x) tuples (unordered)
            shape_type: Predicted shape type ('line', 'rectangle', 'single_cell', etc.)
            mask: Binary mask of the shape instance

        Returns:
            Ordered list of (y, x) tuples
        """
        if len(vertices) == 0:
            return []

        vertices = np.array(vertices)

        # Single vertex or single cell - no ordering needed
        if len(vertices) <= 1 or shape_type == 'single_cell':
            return vertices.tolist()

        # Line: order by distance (endpoints first)
        if shape_type == 'line':
            if len(vertices) == 2:
                return vertices.tolist()
            elif len(vertices) > 2:
                # Find the two most distant points (endpoints)
                max_dist = 0
                best_pair = (0, 1)
                for i in range(len(vertices)):
                    for j in range(i + 1, len(vertices)):
                        dist = np.linalg.norm(vertices[i] - vertices[j])
                        if dist > max_dist:
                            max_dist = dist
                            best_pair = (i, j)
                return [vertices[best_pair[0]].tolist(), vertices[best_pair[1]].tolist()]

        # Rectangle: order vertices by angle around centroid (clockwise or counter-clockwise)
        if shape_type == 'rectangle':
            if len(vertices) < 4:
                # Not enough vertices, fall back to angular ordering
                return self._order_vertices_by_angle(vertices)

            # Take first 4 vertices if there are more
            vertices = vertices[:4]

            # Order by angle around centroid
            return self._order_vertices_by_angle(vertices)

        # Default: order by angle around centroid
        return self._order_vertices_by_angle(vertices)

    def _order_vertices_by_angle(self, vertices):
        """
        Order vertices by angle around their centroid.
        This creates a consistent clockwise or counter-clockwise ordering.

        Args:
            vertices: numpy array of shape (N, 2) with (y, x) coordinates

        Returns:
            List of ordered (y, x) tuples
        """
        if len(vertices) <= 1:
            return vertices.tolist()

        # Compute centroid
        centroid = vertices.mean(axis=0)

        # Compute angle from centroid to each vertex
        # Using atan2(y - cy, x - cx) to get angle in standard mathematical convention
        angles = np.arctan2(vertices[:, 0] - centroid[0],
                           vertices[:, 1] - centroid[1])

        # Sort vertices by angle
        sorted_indices = np.argsort(angles)
        ordered_vertices = vertices[sorted_indices]

        return ordered_vertices.tolist()

    def _trace_perimeter(self, mask):
        """Trace perimeter of a shape."""
        try:
            contours = find_contours(mask.astype(float), 0.5)
            if len(contours) == 0:
                return []

            # Use longest contour
            contour = max(contours, key=len)

            # Simplify
            simplified = approximate_polygon(contour, tolerance=1.0)
            return simplified.tolist()
        except:
            return []

    def _is_filled(self, mask, edge_map):
        """Determine if shape is filled or just outline."""
        if mask.sum() < 4:
            return False

        # If most of the mask is edges, it's an outline
        edge_ratio = (mask * (edge_map > 0.5)).sum() / (mask.sum() + 1e-6)
        return edge_ratio < 0.7

    def _compute_triangle_angles(self, vertices):
        """Compute angles of a triangle."""
        if len(vertices) < 3:
            return []

        angles = []
        for i in range(3):
            v1 = np.array(vertices[i])
            v2 = np.array(vertices[(i+1) % 3])
            v3 = np.array(vertices[(i+2) % 3])

            vec1 = v1 - v2
            vec2 = v3 - v2

            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            angles.append(round(angle, 1))

        return angles

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

    def _fit_circle(self, vertices):
        """Fit circle parameters to vertices."""
        if len(vertices) < 3:
            return {}

        points = np.array(vertices)
        center = points.mean(axis=0)
        radius = np.mean([np.linalg.norm(p - center) for p in points])

        return {
            'center': center.tolist(),
            'radius': round(radius, 2)
        }


# Alias for drop-in replacement
ShapeInference = ShapeInferenceWithClassification

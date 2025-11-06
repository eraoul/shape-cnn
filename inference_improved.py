"""
Improved Inference Pipeline for Shape Recognition
More robust shape classification that handles imperfect vertex detection
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.measure import find_contours, approximate_polygon


class ImprovedShapeInference:
    """Improved inference pipeline with robust shape classification."""

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
        """Predict shapes from a grid."""
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

        # Extract structured information
        objects = self._extract_objects(grid, instance_map, vertex_map, edge_map)

        return {
            'grid_size': (h, w),
            'objects': objects,
            'raw_outputs': {
                'instance_map': instance_map,
                'vertex_map': vertex_map,
                'edge_map': edge_map
            }
        }

    def _extract_objects(self, grid, instance_map, vertex_map, edge_map):
        """Extract structured object information."""
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

            # Trace perimeter
            perimeter = self._trace_perimeter(mask)

            # IMPROVED: Classify shape using multiple signals
            shape_type = self._classify_shape_improved(vertices, mask, cells)

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
            elif shape_type == 'circle':
                obj['circle_params'] = self._fit_circle(vertices)

            objects.append(obj)

        return objects

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

    def _classify_shape_improved(self, vertices, mask, cells):
        """
        Improved shape classification using multiple signals.
        More robust to imperfect vertex detection.
        """
        n = len(vertices)
        area = len(cells)

        # Check if it's a line (elongated shape with few cells)
        if self._is_line_like(mask, vertices, area):
            return 'line'

        # Check if it's circular FIRST (before vertex count)
        if self._is_circle_like(mask):
            return 'circle'

        # Now check vertex-based classification with relaxed constraints
        if n == 0:
            # No vertices detected - use shape analysis
            if self._is_circle_like(mask):
                return 'circle'
            return 'unknown'

        elif n <= 2:
            # Line-like (might have detected 1-2 vertices)
            return 'line'

        elif n == 3 or (n == 4 and not self._is_rectangular_shape(mask)):
            # Triangle (allow 3-4 vertices if not rectangular)
            return 'triangle'

        elif n == 4 or n == 5:
            # Rectangle/quadrilateral (allow 4-5 vertices)
            if self._is_rectangular_shape(mask):
                return 'rectangle'
            return 'quadrilateral'

        else:
            # Many vertices
            if self._is_circle_like(mask):
                return 'circle'
            return 'irregular'

    def _is_line_like(self, mask, vertices, area):
        """Check if shape is line-like based on aspect ratio and area."""
        if area < 2:
            return True

        # Get bounding box
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return False

        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)

        height = max_y - min_y + 1
        width = max_x - min_x + 1

        # Line if very thin (aspect ratio > 3) and small area
        if height == 1 or width == 1:
            return True

        aspect_ratio = max(height, width) / (min(height, width) + 1e-6)

        # Elongated shape with small cross-section
        return aspect_ratio > 3 and area < max(height, width) * 2

    def _is_rectangular_shape(self, mask):
        """Check if mask has rectangular shape (without relying on vertices)."""
        # Get bounding box
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return False

        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)

        # Get bounding box area
        bbox_area = (max_y - min_y + 1) * (max_x - min_x + 1)
        shape_area = mask.sum()

        # Rectangle if fills most of its bounding box
        fill_ratio = shape_area / (bbox_area + 1e-6)

        return fill_ratio > 0.7

    def _is_circle_like(self, mask):
        """Check if mask is circular."""
        # Compute compactness: 4π*area/perimeter²
        area = mask.sum()

        if area < 4:
            return False

        try:
            contours = find_contours(mask.astype(float), 0.5)
            if len(contours) == 0:
                return False
            perimeter = len(contours[0])
            compactness = 4 * np.pi * area / (perimeter ** 2 + 1e-6)

            # Also check if roughly circular (aspect ratio close to 1)
            coords = np.argwhere(mask)
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            height = max_y - min_y + 1
            width = max_x - min_x + 1
            aspect_ratio = max(height, width) / (min(height, width) + 1e-6)

            # Circular if compact AND roughly square
            return compactness > 0.65 and aspect_ratio < 1.5
        except:
            return False

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
ShapeInference = ImprovedShapeInference

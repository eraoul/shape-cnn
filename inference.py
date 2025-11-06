"""
Inference Pipeline for Shape Recognition
Extracts structured information (vertices, edges, perimeters) from grids
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.measure import find_contours, approximate_polygon


class ShapeInference:
    """Inference pipeline for shape recognition."""

    def __init__(self, model, device=None):
        if device is None:
            # Prefer MPS (Apple Silicon) > CUDA > CPU
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
        Predict shapes from a grid.

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

            # Classify shape type
            shape_type = self._classify_shape(vertices, mask)

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
            if shape_type == 'triangle' and len(vertices) == 3:
                obj['angles'] = self._compute_triangle_angles(vertices)
            elif shape_type == 'rectangle' and len(vertices) == 4:
                obj['dimensions'] = self._compute_rectangle_dimensions(vertices)
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

    def _classify_shape(self, vertices, mask):
        """Classify shape based on vertices and mask."""
        n = len(vertices)

        if n == 0:
            return 'unknown'
        elif n == 2:
            return 'line'
        elif n == 3:
            return 'triangle'
        elif n == 4:
            if self._is_rectangle(vertices):
                return 'rectangle'
            return 'quadrilateral'
        elif self._is_circle_like(mask):
            return 'circle'
        else:
            return 'irregular'

    def _is_rectangle(self, vertices):
        """Check if vertices form a rectangle."""
        if len(vertices) != 4:
            return False

        # Compute angles
        angles = []
        for i in range(4):
            v1 = np.array(vertices[i])
            v2 = np.array(vertices[(i+1) % 4])
            v3 = np.array(vertices[(i+2) % 4])

            vec1 = v1 - v2
            vec2 = v3 - v2

            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            angles.append(angle)

        # Check if all angles are close to 90 degrees
        return all(abs(angle - 90) < 20 for angle in angles)

    def _is_circle_like(self, mask):
        """Check if mask is circular."""
        # Compute compactness: 4π*area/perimeter²
        area = mask.sum()
        try:
            contours = find_contours(mask.astype(float), 0.5)
            if len(contours) == 0:
                return False
            perimeter = len(contours[0])
            compactness = 4 * np.pi * area / (perimeter ** 2)
            return compactness > 0.7  # Circles have compactness close to 1
        except:
            return False

    def _is_filled(self, mask, edge_map):
        """Determine if shape is filled or just outline."""
        if mask.sum() < 4:
            return False

        # If most of the mask is edges, it's an outline
        edge_ratio = (mask * (edge_map > 0.5)).sum() / mask.sum()
        return edge_ratio < 0.7

    def _compute_triangle_angles(self, vertices):
        """Compute angles of a triangle."""
        if len(vertices) != 3:
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
        if len(vertices) != 4:
            return {}

        v = np.array(vertices)
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

    def visualize(self, grid, prediction, save_path=None):
        """Visualize prediction results."""
        h, w = grid.shape[:2]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Helper function to add gridlines
        def add_gridlines(ax, h, w):
            import numpy as np
            for x in np.arange(0.5, w, 1):
                ax.axvline(x, color='white', linewidth=0.5, alpha=0.3)
            for y in np.arange(0.5, h, 1):
                ax.axhline(y, color='white', linewidth=0.5, alpha=0.3)

        # Input grid
        input_rgb = grid.argmax(axis=2)
        axes[0, 0].imshow(input_rgb, cmap='tab10', interpolation='nearest')
        add_gridlines(axes[0, 0], h, w)
        axes[0, 0].set_title('Input Grid')
        axes[0, 0].axis('off')

        # Instance segmentation
        axes[0, 1].imshow(prediction['raw_outputs']['instance_map'], cmap='tab10', interpolation='nearest')
        add_gridlines(axes[0, 1], h, w)
        axes[0, 1].set_title('Instance Segmentation')
        axes[0, 1].axis('off')

        # Vertex heatmap
        axes[0, 2].imshow(prediction['raw_outputs']['vertex_map'], cmap='hot', interpolation='nearest')
        add_gridlines(axes[0, 2], h, w)
        axes[0, 2].set_title('Vertex Heatmap')
        axes[0, 2].axis('off')

        # Edge heatmap
        axes[1, 0].imshow(prediction['raw_outputs']['edge_map'], cmap='hot', interpolation='nearest')
        add_gridlines(axes[1, 0], h, w)
        axes[1, 0].set_title('Edge Heatmap')
        axes[1, 0].axis('off')

        # Detected vertices
        axes[1, 1].imshow(input_rgb, cmap='tab10', alpha=0.5, interpolation='nearest')
        add_gridlines(axes[1, 1], h, w)
        for obj in prediction['objects']:
            for vy, vx in obj['vertices']:
                axes[1, 1].plot(vx, vy, 'r*', markersize=15)
        axes[1, 1].set_title('Detected Vertices')
        axes[1, 1].axis('off')

        # Structured output
        axes[1, 2].axis('off')
        text = f"Grid: {prediction['grid_size']}\n"
        text += f"Objects: {len(prediction['objects'])}\n\n"
        for obj in prediction['objects']:
            text += f"• {obj['type'].capitalize()}\n"
            text += f"  Color: {obj['color']}, "
            text += f"Verts: {obj['properties']['num_vertices']}\n"
        axes[1, 2].text(0.1, 0.9, text, fontsize=10, verticalalignment='top',
                       fontfamily='monospace')
        axes[1, 2].set_title('Structured Output')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

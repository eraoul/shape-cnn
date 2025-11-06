"""
Utility Functions for Shape Recognition Pipeline
Contains collate function for variable-sized grids and helper utilities
"""

import torch
import torch.nn.functional as F


def collate_variable_size(batch):
    """
    Custom collate function for variable-sized grids.

    Pads all samples in a batch to the maximum height and width,
    allowing batching of grids with different dimensions.

    Args:
        batch: List of samples from ShapeDataset

    Returns:
        Dictionary with batched tensors and original sizes
    """
    # Find max dimensions in batch
    max_h = max(item['grid'].shape[1] for item in batch)
    max_w = max(item['grid'].shape[2] for item in batch)
    num_colors = batch[0]['grid'].shape[0]

    # Check if shape_type_map is present (for new dataset with classification)
    has_shape_type_map = 'shape_type_map' in batch[0]

    # Pad all samples to max size
    padded_batch = {
        'grid': [],
        'instance_map': [],
        'vertex_map': [],
        'edge_map': [],
        'original_sizes': []
    }

    if has_shape_type_map:
        padded_batch['shape_type_map'] = []

    for item in batch:
        c, h, w = item['grid'].shape
        pad_h, pad_w = max_h - h, max_w - w

        # Pad tensors
        grid = F.pad(item['grid'], (0, pad_w, 0, pad_h))
        instance_map = F.pad(item['instance_map'], (0, pad_w, 0, pad_h))
        vertex_map = F.pad(item['vertex_map'], (0, pad_w, 0, pad_h))
        edge_map = F.pad(item['edge_map'], (0, pad_w, 0, pad_h))

        padded_batch['grid'].append(grid)
        padded_batch['instance_map'].append(instance_map)
        padded_batch['vertex_map'].append(vertex_map)
        padded_batch['edge_map'].append(edge_map)
        padded_batch['original_sizes'].append((h, w))

        # Pad shape_type_map if present
        if has_shape_type_map:
            shape_type_map = F.pad(item['shape_type_map'], (0, pad_w, 0, pad_h))
            padded_batch['shape_type_map'].append(shape_type_map)

    result = {
        'grid': torch.stack(padded_batch['grid']),
        'instance_map': torch.stack(padded_batch['instance_map']),
        'vertex_map': torch.stack(padded_batch['vertex_map']),
        'edge_map': torch.stack(padded_batch['edge_map']),
        'original_sizes': padded_batch['original_sizes']
    }

    if has_shape_type_map:
        result['shape_type_map'] = torch.stack(padded_batch['shape_type_map'])

    return result


def collate_keypoint_dataset(batch):
    """
    Custom collate function for keypoint dataset with variable-sized grids.

    Pads all samples in a batch to the maximum height and width.
    Handles keypoint targets instead of vertex/edge maps.

    Args:
        batch: List of samples from ShapeDatasetWithKeypoints

    Returns:
        Dictionary with batched tensors, original sizes, and keypoint targets
    """
    # Find max dimensions in batch
    max_h = max(item['grid'].shape[1] for item in batch)
    max_w = max(item['grid'].shape[2] for item in batch)

    # Pad all samples to max size
    padded_batch = {
        'grid': [],
        'instance_map': [],
        'shape_type_map': [],
        'keypoint_targets': [],
        'original_sizes': []
    }

    for item in batch:
        c, h, w = item['grid'].shape
        pad_h, pad_w = max_h - h, max_w - w

        # Pad spatial tensors
        grid = F.pad(item['grid'], (0, pad_w, 0, pad_h))
        instance_map = F.pad(item['instance_map'], (0, pad_w, 0, pad_h))
        shape_type_map = F.pad(item['shape_type_map'], (0, pad_w, 0, pad_h))

        padded_batch['grid'].append(grid)
        padded_batch['instance_map'].append(instance_map)
        padded_batch['shape_type_map'].append(shape_type_map)
        padded_batch['keypoint_targets'].append(item['keypoint_targets'])
        padded_batch['original_sizes'].append(item['original_sizes'])

    result = {
        'grid': torch.stack(padded_batch['grid']),
        'instance_map': torch.stack(padded_batch['instance_map']),
        'shape_type_map': torch.stack(padded_batch['shape_type_map']),
        'keypoint_targets': torch.stack(padded_batch['keypoint_targets']),
        'original_sizes': padded_batch['original_sizes']
    }

    return result

"""
Analyze Keypoint Model Performance
Evaluates keypoint accuracy and shape classification
"""

import torch
import numpy as np
from data_generation_with_keypoints import ShapeGeneratorWithKeypoints
from model_with_keypoints import ShapeNetWithKeypoints
from inference_with_keypoints import KeypointInference


def analyze_keypoint_performance(num_samples=100, model_path='best_model_keypoints.pth'):
    """Analyze keypoint model performance on random samples."""

    print("="*80)
    print("KEYPOINT MODEL PERFORMANCE ANALYSIS")
    print("="*80)

    # Load model
    print("\nLoading model...")
    model = ShapeNetWithKeypoints(num_colors=10, max_instances=10)
    model.load_state_dict(torch.load(model_path))
    inference = KeypointInference(model)
    print(f"Using device: {inference.device}")

    # Generate test samples
    print(f"\nGenerating {num_samples} test samples...")
    generator = ShapeGeneratorWithKeypoints(num_colors=10, min_size=5, max_size=25, max_instances=10)

    # Metrics storage
    metrics = {
        'keypoint_errors': [],  # Distance errors for each keypoint
        'shape_accuracy': [],   # Correct shape classification
        'instance_accuracy': [],  # Instance segmentation accuracy
        'valid_keypoint_accuracy': [],  # Correctly predicted valid keypoints
        'num_vertices_correct': []  # Correct number of vertices detected
    }

    shape_type_metrics = {
        'line': {'errors': [], 'count': 0},
        'rectangle': {'errors': [], 'count': 0},
        'single_cell': {'errors': [], 'count': 0}
    }

    for sample_idx in range(num_samples):
        sample = generator.generate_sample()
        grid = sample['grid']
        h, w = sample['grid_size']

        # Ground truth
        gt_keypoints = sample['keypoint_targets']  # [max_instances, max_keypoints, 3]
        gt_instance_map = sample['instance_map']
        gt_shape_map = sample['shape_type_map']

        # Run inference
        prediction = inference.predict(grid)

        # Instance segmentation accuracy
        pred_instance_map = prediction['raw_outputs']['instance_map']
        inst_acc = (pred_instance_map == gt_instance_map).mean()
        metrics['instance_accuracy'].append(inst_acc)

        # Shape classification accuracy
        pred_shape_map = prediction['raw_outputs']['shape_class_map']
        # Only evaluate on non-background pixels
        non_bg_mask = gt_shape_map > 0
        if non_bg_mask.sum() > 0:
            shape_acc = (pred_shape_map[non_bg_mask] == gt_shape_map[non_bg_mask]).mean()
            metrics['shape_accuracy'].append(shape_acc)

        # Keypoint accuracy
        pred_keypoints = prediction['raw_outputs']['keypoints']  # [max_instances, max_keypoints, 3]

        # Analyze each instance
        for inst_id in range(1, 11):  # Instances 1-10
            inst_idx = inst_id - 1

            # Check if this instance exists in ground truth
            gt_has_instance = (gt_instance_map == inst_id).sum() > 0

            if not gt_has_instance:
                continue

            # Get shape type from metadata
            if 'shapes' in sample['metadata'] and inst_idx < len(sample['metadata']['shapes']):
                shape_type = sample['metadata']['shapes'][inst_idx]['type']
            elif 'type' in sample['metadata']:
                shape_type = sample['metadata']['type']
            else:
                continue

            shape_type_metrics[shape_type]['count'] += 1

            # Expected number of keypoints
            num_expected = ShapeNetWithKeypoints.get_num_keypoints(shape_type)

            # Extract valid GT keypoints
            gt_kp = gt_keypoints[inst_idx]
            gt_valid = gt_kp[:, 2] > 0.5
            gt_coords = gt_kp[gt_valid, :2]  # Normalized coords

            # Extract valid predicted keypoints
            pred_kp = pred_keypoints[inst_idx]
            pred_valid = pred_kp[:, 2] > 0.5
            pred_coords = pred_kp[pred_valid, :2]  # Normalized coords

            # Check if correct number of keypoints detected
            num_correct = (gt_valid.sum() == pred_valid.sum())
            metrics['num_vertices_correct'].append(float(num_correct))

            # Validity accuracy
            if num_expected > 0:
                valid_acc = (gt_valid[:num_expected] == pred_valid[:num_expected]).mean()
                metrics['valid_keypoint_accuracy'].append(valid_acc)

            # Compute keypoint distance errors
            for i in range(min(len(gt_coords), len(pred_coords))):
                # Denormalize to pixel coordinates
                gt_y, gt_x = gt_coords[i, 0] * h, gt_coords[i, 1] * w
                pred_y, pred_x = pred_coords[i, 0] * h, pred_coords[i, 1] * w

                # Euclidean distance
                error = np.sqrt((gt_y - pred_y)**2 + (gt_x - pred_x)**2)
                metrics['keypoint_errors'].append(error)
                shape_type_metrics[shape_type]['errors'].append(error)

    # Print summary
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)

    print(f"\nInstance Segmentation Accuracy: {np.mean(metrics['instance_accuracy']):.3f}")
    print(f"Shape Classification Accuracy: {np.mean(metrics['shape_accuracy']):.3f}")
    print(f"Valid Keypoint Accuracy: {np.mean(metrics['valid_keypoint_accuracy']):.3f}")
    print(f"Correct Vertex Count: {np.mean(metrics['num_vertices_correct']):.3f}")

    print(f"\nKeypoint Localization Error:")
    errors = np.array(metrics['keypoint_errors'])
    print(f"  Mean: {errors.mean():.2f} pixels")
    print(f"  Median: {np.median(errors):.2f} pixels")
    print(f"  Std: {errors.std():.2f} pixels")
    print(f"  Max: {errors.max():.2f} pixels")
    print(f"  < 1 pixel: {(errors < 1.0).mean()*100:.1f}%")
    print(f"  < 2 pixels: {(errors < 2.0).mean()*100:.1f}%")
    print(f"  < 3 pixels: {(errors < 3.0).mean()*100:.1f}%")

    # Per-shape analysis
    print("\n" + "="*80)
    print("PER-SHAPE TYPE ANALYSIS")
    print("="*80)

    for shape_type in ['line', 'rectangle', 'single_cell']:
        shape_data = shape_type_metrics[shape_type]
        if shape_data['count'] > 0:
            print(f"\n{shape_type.upper()}:")
            print(f"  Count: {shape_data['count']}")
            if len(shape_data['errors']) > 0:
                shape_errors = np.array(shape_data['errors'])
                print(f"  Mean error: {shape_errors.mean():.2f} pixels")
                print(f"  Median error: {np.median(shape_errors):.2f} pixels")
                print(f"  < 2 pixels: {(shape_errors < 2.0).mean()*100:.1f}%")

    print("\n" + "="*80)


if __name__ == "__main__":
    import sys

    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'best_model_keypoints.pth'

    analyze_keypoint_performance(num_samples, model_path)

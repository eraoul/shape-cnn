"""
Analyze Model Performance by Shape Type
Helps identify which shapes the model struggles with
"""

import torch
import numpy as np
from data_generation import ShapeDataset
from model import StructuralShapeNet
from inference import ShapeInference
from collections import defaultdict


def analyze_by_shape_type(dataset_path, model_path='best_model.pth', num_colors=8, max_instances=10):
    """
    Analyze model performance broken down by shape type.
    """
    print("="*80)
    print("SHAPE RECOGNITION PERFORMANCE ANALYSIS")
    print("="*80)

    # Load dataset and model
    print("\nLoading dataset and model...")
    # When loading from cache, num_samples is ignored and loaded from file
    dataset = ShapeDataset(num_samples=5000, num_colors=num_colors, cache_path=dataset_path)
    model = StructuralShapeNet(num_colors, max_instances)
    model.load_state_dict(torch.load(model_path))
    inference = ShapeInference(model)

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Using device: {inference.device}")

    # Track statistics by shape type
    stats = defaultdict(lambda: {
        'total': 0,
        'vertex_correct': 0,
        'vertex_close': 0,
        'shape_detected': 0,
        'vertex_errors': [],
        'sample_indices': []
    })

    print("\nAnalyzing samples...")
    for idx in range(len(dataset)):
        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(dataset)} samples...")

        sample = dataset.get_raw_sample(idx)
        test_grid = sample['grid']

        # Get ground truth shape type from metadata
        if 'metadata' not in sample:
            continue

        meta = sample['metadata']

        # Handle single shape or multiple shapes
        if 'type' in meta:
            shape_types = [meta['type']]
        elif 'shapes' in meta:
            shape_types = [s['type'] for s in meta['shapes']]
        else:
            continue

        # Run inference
        prediction = inference.predict(test_grid)

        # Get ground truth vertices
        gt_vertex = sample['vertex_map']
        gt_vertices = np.argwhere(gt_vertex > 0.5)

        # For each ground truth shape
        for shape_type in shape_types:
            stats[shape_type]['total'] += 1
            stats[shape_type]['sample_indices'].append(idx)

            # Check if shape was detected
            detected_types = [obj['type'] for obj in prediction['objects']]
            if shape_type in detected_types or (shape_type == 'rectangle' and 'quadrilateral' in detected_types):
                stats[shape_type]['shape_detected'] += 1

            # Check vertex detection accuracy
            pred_vertex = prediction['raw_outputs']['vertex_map']
            pred_vertices = np.argwhere(pred_vertex > 0.3)

            if len(gt_vertices) > 0:
                # Calculate vertex detection accuracy
                correct = 0
                close = 0

                for gv in gt_vertices:
                    # Check if any predicted vertex is close
                    distances = [np.linalg.norm(gv - pv) for pv in pred_vertices] if len(pred_vertices) > 0 else [999]
                    min_dist = min(distances)

                    if min_dist < 1.0:
                        correct += 1
                    elif min_dist < 2.0:
                        close += 1

                vertex_acc = correct / len(gt_vertices)
                stats[shape_type]['vertex_correct'] += vertex_acc
                stats[shape_type]['vertex_close'] += (correct + close) / len(gt_vertices)
                stats[shape_type]['vertex_errors'].append(1.0 - vertex_acc)

    # Print results
    print("\n" + "="*80)
    print("RESULTS BY SHAPE TYPE")
    print("="*80)

    shape_order = ['single_cell', 'line', 'triangle', 'rectangle', 'circle', 'irregular']

    for shape_type in shape_order:
        if shape_type not in stats or stats[shape_type]['total'] == 0:
            continue

        s = stats[shape_type]
        print(f"\n{shape_type.upper()}")
        print(f"  Total samples: {s['total']}")
        print(f"  Shape detection rate: {s['shape_detected']/s['total']*100:.1f}%")
        print(f"  Avg vertex accuracy (strict): {s['vertex_correct']/s['total']*100:.1f}%")
        print(f"  Avg vertex accuracy (relaxed): {s['vertex_close']/s['total']*100:.1f}%")

        if s['vertex_errors']:
            avg_error = np.mean(s['vertex_errors'])
            print(f"  Avg vertex error: {avg_error*100:.1f}%")

        # Find worst examples
        if len(s['vertex_errors']) > 0:
            worst_indices = np.argsort(s['vertex_errors'])[-3:][::-1]
            worst_samples = [s['sample_indices'][i] for i in worst_indices if i < len(s['sample_indices'])]
            print(f"  Worst examples (indices): {worst_samples}")

    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)

    total_samples = sum(s['total'] for s in stats.values())
    total_detected = sum(s['shape_detected'] for s in stats.values())
    total_vertex_acc = sum(s['vertex_correct'] for s in stats.values())

    print(f"Total samples analyzed: {total_samples}")
    print(f"Overall shape detection rate: {total_detected/total_samples*100:.1f}%")
    print(f"Overall vertex accuracy: {total_vertex_acc/total_samples*100:.1f}%")

    return stats


def find_failing_examples(dataset_path, model_path='best_model.pth', shape_type='rectangle', num_examples=10):
    """
    Find specific examples where the model fails on a given shape type.
    """
    print(f"\nFinding failing examples for {shape_type}...")

    # When loading from cache, num_samples is ignored and loaded from file
    dataset = ShapeDataset(num_samples=5000, num_colors=8, cache_path=dataset_path)
    model = StructuralShapeNet(8, 10)
    model.load_state_dict(torch.load(model_path))
    inference = ShapeInference(model)

    failing_examples = []

    for idx in range(len(dataset)):
        sample = dataset.get_raw_sample(idx)

        # Check if this is the shape type we're looking for
        if 'metadata' not in sample:
            continue

        meta = sample['metadata']
        sample_shape_type = meta.get('type', None)

        if sample_shape_type != shape_type:
            if 'shapes' in meta:
                shape_types = [s['type'] for s in meta['shapes']]
                if shape_type not in shape_types:
                    continue
            else:
                continue

        # Run inference
        test_grid = sample['grid']
        prediction = inference.predict(test_grid)

        # Check if prediction is wrong
        detected_types = [obj['type'] for obj in prediction['objects']]

        # Check vertex accuracy
        gt_vertex = sample['vertex_map']
        pred_vertex = prediction['raw_outputs']['vertex_map']

        gt_vertices = np.argwhere(gt_vertex > 0.5)
        pred_vertices = np.argwhere(pred_vertex > 0.3)

        if len(gt_vertices) > 0:
            correct = 0
            for gv in gt_vertices:
                distances = [np.linalg.norm(gv - pv) for pv in pred_vertices] if len(pred_vertices) > 0 else [999]
                if min(distances) < 1.5:
                    correct += 1

            accuracy = correct / len(gt_vertices)

            # If accuracy is low or shape not detected, add to failing examples
            if accuracy < 0.5 or shape_type not in detected_types:
                failing_examples.append((idx, accuracy))

        if len(failing_examples) >= num_examples:
            break

    print(f"\nFound {len(failing_examples)} failing examples:")
    for idx, acc in failing_examples:
        print(f"  Sample {idx}: vertex accuracy = {acc*100:.1f}%")

    return failing_examples


if __name__ == "__main__":
    import sys

    dataset_path = 'data/train_dataset_large.pkl'
    model_path = 'best_model.pth'

    if len(sys.argv) > 1 and sys.argv[1] == '--find-failures':
        shape_type = sys.argv[2] if len(sys.argv) > 2 else 'rectangle'
        failing = find_failing_examples(dataset_path, model_path, shape_type)
    else:
        # Run full analysis
        stats = analyze_by_shape_type(dataset_path, model_path)

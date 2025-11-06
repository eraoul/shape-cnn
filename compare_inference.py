"""
Compare Original vs Improved Inference
Shows the difference in shape classification accuracy
"""

import torch
import numpy as np
from data_generation import ShapeDataset
from model import StructuralShapeNet
from inference import ShapeInference as OriginalInference
from inference_improved import ImprovedShapeInference
from collections import defaultdict


def compare_inference_methods(dataset_path, model_path, num_samples=500):
    """Compare original vs improved inference."""

    print("="*80)
    print("COMPARING INFERENCE METHODS")
    print("="*80)

    # Load
    print("\nLoading dataset and model...")
    dataset = ShapeDataset(num_samples=5000, num_colors=8, cache_path=dataset_path)
    model = StructuralShapeNet(8, 10)
    model.load_state_dict(torch.load(model_path))

    original_inference = OriginalInference(model)
    improved_inference = ImprovedShapeInference(model)

    print(f"Loaded {len(dataset)} samples")
    print(f"Testing on first {num_samples} samples...")

    # Stats
    original_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    improved_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    for idx in range(min(num_samples, len(dataset))):
        if idx % 100 == 0 and idx > 0:
            print(f"  Processed {idx}/{num_samples}...")

        sample = dataset.get_raw_sample(idx)
        test_grid = sample['grid']

        # Get ground truth
        if 'metadata' not in sample:
            continue

        meta = sample['metadata']
        if 'type' in meta:
            gt_types = [meta['type']]
        elif 'shapes' in meta:
            gt_types = [s['type'] for s in meta['shapes']]
        else:
            continue

        # Run both inferences
        original_pred = original_inference.predict(test_grid)
        improved_pred = improved_inference.predict(test_grid)

        original_types = [obj['type'] for obj in original_pred['objects']]
        improved_types = [obj['type'] for obj in improved_pred['objects']]

        # Check accuracy for each ground truth shape
        for gt_type in gt_types:
            original_stats[gt_type]['total'] += 1
            improved_stats[gt_type]['total'] += 1

            # Allow some flexibility in matching (rectangle = quadrilateral)
            if gt_type in original_types:
                original_stats[gt_type]['correct'] += 1
            elif gt_type == 'rectangle' and 'quadrilateral' in original_types:
                original_stats[gt_type]['correct'] += 1

            if gt_type in improved_types:
                improved_stats[gt_type]['correct'] += 1
            elif gt_type == 'rectangle' and 'quadrilateral' in improved_types:
                improved_stats[gt_type]['correct'] += 1

    # Print comparison
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)

    shape_types = ['line', 'triangle', 'rectangle', 'circle', 'irregular']

    print(f"\n{'Shape':<15} {'Original':<15} {'Improved':<15} {'Improvement':<15}")
    print("-"*60)

    overall_original_correct = 0
    overall_original_total = 0
    overall_improved_correct = 0
    overall_improved_total = 0

    for shape_type in shape_types:
        if original_stats[shape_type]['total'] == 0:
            continue

        orig_acc = original_stats[shape_type]['correct'] / original_stats[shape_type]['total']
        impr_acc = improved_stats[shape_type]['correct'] / improved_stats[shape_type]['total']
        improvement = impr_acc - orig_acc

        overall_original_correct += original_stats[shape_type]['correct']
        overall_original_total += original_stats[shape_type]['total']
        overall_improved_correct += improved_stats[shape_type]['correct']
        overall_improved_total += improved_stats[shape_type]['total']

        print(f"{shape_type:<15} {orig_acc*100:>6.1f}%        {impr_acc*100:>6.1f}%        {improvement*100:>+6.1f}%")

    print("-"*60)

    overall_orig = overall_original_correct / overall_original_total
    overall_impr = overall_improved_correct / overall_improved_total
    overall_improvement = overall_impr - overall_orig

    print(f"{'OVERALL':<15} {overall_orig*100:>6.1f}%        {overall_impr*100:>6.1f}%        {overall_improvement*100:>+6.1f}%")

    print("\n" + "="*80)

    if overall_improvement > 0:
        print(f"✓ Improved inference is {overall_improvement*100:.1f}% better!")
    else:
        print(f"✗ Improved inference is {abs(overall_improvement)*100:.1f}% worse.")

    return original_stats, improved_stats


if __name__ == "__main__":
    import sys

    dataset_path = 'data/train_dataset_large.pkl'
    model_path = 'best_model.pth'

    num_samples = 500
    if len(sys.argv) > 1:
        num_samples = int(sys.argv[1])

    compare_inference_methods(dataset_path, model_path, num_samples)

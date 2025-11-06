"""
Analyze Learned Classification Performance
Evaluates how well the neural network learned to classify shapes
"""

import torch
import numpy as np
from data_generation_with_types import ShapeDatasetWithTypes
from model_small import ShapeNetWithClassification
from model_large import LargeShapeNet
from inference_with_classification import ShapeInferenceWithClassification
from collections import defaultdict


def analyze_classification_performance(dataset_path, model_path, num_samples=1000, model_class=None):
    """
    Analyze shape classification performance.
    """
    print("="*80)
    print("LEARNED SHAPE CLASSIFICATION PERFORMANCE ANALYSIS")
    print("="*80)

    # Auto-detect model class if not provided
    if model_class is None:
        if 'large' in model_path:
            model_class = LargeShapeNet
            print("Auto-detected: LargeShapeNet")
        else:
            model_class = ShapeNetWithClassification
            print("Auto-detected: ShapeNetWithClassification")

    # Load
    print("\nLoading dataset and model...")
    NUM_COLORS = 10  # Must match training configuration
    dataset = ShapeDatasetWithTypes(num_samples=5000, num_colors=NUM_COLORS, cache_path=dataset_path)
    model = model_class(NUM_COLORS, 10)
    model.load_state_dict(torch.load(model_path))
    inference = ShapeInferenceWithClassification(model)

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Analyzing first {num_samples} samples...")
    print(f"Using device: {inference.device}")

    # Track statistics
    stats = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'confused_with': defaultdict(int)
    })

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

        # Run inference
        prediction = inference.predict(test_grid)
        pred_types = [obj['type'] for obj in prediction['objects']]

        # Check accuracy for each ground truth shape
        for gt_type in gt_types:
            stats[gt_type]['total'] += 1

            if gt_type in pred_types:
                stats[gt_type]['correct'] += 1
            else:
                # Track confusion
                for pred_type in pred_types:
                    if pred_type != gt_type:
                        stats[gt_type]['confused_with'][pred_type] += 1

    # Print results
    print("\n" + "="*80)
    print("CLASSIFICATION ACCURACY BY SHAPE TYPE")
    print("="*80)

    shape_types = ['single_cell', 'line', 'triangle', 'rectangle', 'circle', 'irregular']

    print(f"\n{'Shape':<15} {'Total':<10} {'Correct':<10} {'Accuracy':<10}")
    print("-"*50)

    overall_correct = 0
    overall_total = 0

    for shape_type in shape_types:
        if stats[shape_type]['total'] == 0:
            continue

        total = stats[shape_type]['total']
        correct = stats[shape_type]['correct']
        accuracy = correct / total

        overall_correct += correct
        overall_total += total

        print(f"{shape_type:<15} {total:<10} {correct:<10} {accuracy*100:>6.1f}%")

        # Show top confusions
        if stats[shape_type]['confused_with']:
            confusions = sorted(stats[shape_type]['confused_with'].items(),
                              key=lambda x: x[1], reverse=True)[:2]
            conf_str = ", ".join([f"{pred}({count})" for pred, count in confusions])
            print(f"                Confused with: {conf_str}")

    print("-"*50)
    overall_acc = overall_correct / overall_total
    print(f"{'OVERALL':<15} {overall_total:<10} {overall_correct:<10} {overall_acc*100:>6.1f}%")

    print("\n" + "="*80)

    # Compare with expected performance
    print("\nEXPECTED IMPROVEMENTS:")
    print("  - Old model (heuristic): ~12-16% for simple shapes")
    print("  - Old model (heuristic): ~59% for circles")
    print(f"  - New model (learned):   {overall_acc*100:.1f}% overall")

    if overall_acc > 0.50:
        print("\n✓ Learned classification is working well!")
    elif overall_acc > 0.30:
        print("\n⚠ Learned classification shows improvement but needs more training")
    else:
        print("\n✗ Learned classification needs debugging")

    return stats


if __name__ == "__main__":
    import sys

    # Default to large model and large dataset
    dataset_path = 'data/train_dataset_large_50k.pkl'
    model_path = 'best_model_large.pth'

    num_samples = 1000
    if len(sys.argv) > 1:
        num_samples = int(sys.argv[1])

    # Allow specifying model path as second argument
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
        # Auto-adjust dataset path for small model
        if 'small' in model_path or 'large' not in model_path:
            dataset_path = 'data/train_dataset_with_types.pkl'

    stats = analyze_classification_performance(dataset_path, model_path, num_samples)

"""
Compare Different Model Sizes and Approaches
Shows the improvement from larger model and more data
"""

import torch
import numpy as np
from data_generation_with_types import ShapeDatasetWithTypes
from collections import defaultdict


def load_and_test_model(model_class, model_path, dataset_path, num_samples=500):
    """Load a model and test its performance."""
    from inference_with_classification import ShapeInferenceWithClassification

    print(f"\nLoading model from {model_path}...")

    try:
        NUM_COLORS = 10  # Must match training configuration
        model = model_class(NUM_COLORS, 10)
        model.load_state_dict(torch.load(model_path))
        inference = ShapeInferenceWithClassification(model)

        # Load dataset
        dataset = ShapeDatasetWithTypes(num_samples=5000, num_colors=NUM_COLORS,
                                       cache_path=dataset_path)

        print(f"Testing on {num_samples} samples...")

        stats = defaultdict(lambda: {'total': 0, 'correct': 0})

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

            # Check accuracy
            for gt_type in gt_types:
                stats[gt_type]['total'] += 1
                if gt_type in pred_types:
                    stats[gt_type]['correct'] += 1

        return stats

    except FileNotFoundError:
        print(f"  ✗ Model file not found: {model_path}")
        return None
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return None


def compare_models(num_samples=500):
    """Compare different model configurations."""

    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    dataset_path_small = 'data/train_dataset_with_types.pkl'
    dataset_path_large = 'data/train_dataset_large_50k.pkl'

    # Try to test different model sizes
    models_to_test = [
        ("Small Model (10k data)", "model_with_classification", "ShapeNetWithClassification",
         "best_model_small.pth", dataset_path_small),
        ("Large Model (50k data)", "model_large", "LargeShapeNet",
         "best_model_large.pth", dataset_path_large),
    ]

    results = {}

    for name, module_name, class_name, model_path, dataset_path in models_to_test:
        print(f"\n{'-'*80}")
        print(f"Testing: {name}")
        print(f"{'-'*80}")

        try:
            # Dynamically import the model class
            module = __import__(module_name)
            model_class = getattr(module, class_name)

            stats = load_and_test_model(model_class, model_path, dataset_path, num_samples)

            if stats:
                results[name] = stats

        except Exception as e:
            print(f"  ✗ Could not test {name}: {e}")
            continue

    # Print comparison
    if len(results) > 0:
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)

        shape_types = ['line', 'triangle', 'rectangle', 'circle', 'irregular']

        # Print header
        print(f"\n{'Shape':<15}", end='')
        for model_name in results.keys():
            print(f" {model_name:<25}", end='')
        print()
        print("-" * (15 + 25 * len(results)))

        # Print results for each shape
        for shape_type in shape_types:
            print(f"{shape_type:<15}", end='')

            for model_name, stats in results.items():
                if stats[shape_type]['total'] > 0:
                    acc = stats[shape_type]['correct'] / stats[shape_type]['total']
                    print(f" {acc*100:>6.1f}% ({stats[shape_type]['total']:<4})", end='')
                else:
                    print(f" {'N/A':<6} {'':4}", end='')

            print()

        # Print overall
        print("-" * (15 + 25 * len(results)))
        print(f"{'OVERALL':<15}", end='')

        for model_name, stats in results.items():
            total_correct = sum(s['correct'] for s in stats.values())
            total_samples = sum(s['total'] for s in stats.values())
            if total_samples > 0:
                overall_acc = total_correct / total_samples
                print(f" {overall_acc*100:>6.1f}% ({total_samples:<4})", end='')
            else:
                print(f" {'N/A':<6} {'':4}", end='')

        print()

        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        # Find best model
        best_acc = 0
        best_model = None

        for model_name, stats in results.items():
            total_correct = sum(s['correct'] for s in stats.values())
            total_samples = sum(s['total'] for s in stats.values())
            if total_samples > 0:
                acc = total_correct / total_samples
                if acc > best_acc:
                    best_acc = acc
                    best_model = model_name

        if best_model:
            print(f"\nBest model: {best_model} with {best_acc*100:.1f}% accuracy")

            if best_acc > 0.80:
                print("✓ Excellent performance! Model is working well.")
            elif best_acc > 0.60:
                print("✓ Good performance. Consider more training or data augmentation.")
            elif best_acc > 0.40:
                print("⚠ Moderate performance. Try larger model or more data.")
            else:
                print("✗ Low performance. Check model architecture and training.")

    else:
        print("\n✗ No models could be tested. Train a model first:")
        print("  python train_with_classification.py")
        print("  python train_large_model.py")


if __name__ == "__main__":
    import sys

    num_samples = 500
    if len(sys.argv) > 1:
        num_samples = int(sys.argv[1])

    compare_models(num_samples)

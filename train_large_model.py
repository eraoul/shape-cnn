"""
Training Script with Large Model and Large Dataset
Optimized for maximum performance
"""

import torch
from torch.utils.data import DataLoader

from data_generation_with_types import ShapeDatasetWithTypes
from model_large import LargeShapeNet
from training_with_classification import ShapeRecognitionTrainerWithClassification
from inference_with_classification import ShapeInferenceWithClassification
from utils import collate_variable_size


def main():
    """Main training function with large model and dataset."""

    # LARGE CONFIGURATION
    NUM_COLORS = 10
    MAX_INSTANCES = 10
    BATCH_SIZE = 48  # Larger batch size for better gradient estimates
    NUM_EPOCHS = 80  # More epochs
    TRAIN_SAMPLES = 50000  # 5x more training data
    VAL_SAMPLES = 5000  # 5x more validation data

    print("=" * 80)
    print("LARGE-SCALE SHAPE RECOGNITION TRAINING")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  - Training samples: {TRAIN_SAMPLES:,}")
    print(f"  - Validation samples: {VAL_SAMPLES:,}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Model: Large architecture (~2-3M parameters)")
    print("=" * 80)

    # 1. Create large datasets
    print("\n1. Creating/loading large training dataset...")
    print("   (This will take 5-10 minutes on first run, then cached)")

    train_dataset = ShapeDatasetWithTypes(
        TRAIN_SAMPLES, NUM_COLORS, min_size=1, max_size=30,
        cache_path='data/train_dataset_large_50k.pkl'
    )
    val_dataset = ShapeDatasetWithTypes(
        VAL_SAMPLES, NUM_COLORS, min_size=1, max_size=30,
        cache_path='data/val_dataset_large_5k.pkl'
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_variable_size,
                             num_workers=0)  # Set to 0 for MPS compatibility
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, collate_fn=collate_variable_size,
                           num_workers=0)

    print(f"   ✓ Training samples: {TRAIN_SAMPLES:,}")
    print(f"   ✓ Validation samples: {VAL_SAMPLES:,}")

    # 2. Create large model
    print("\n2. Initializing large model...")
    model = LargeShapeNet(NUM_COLORS, MAX_INSTANCES)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

    # 3. Create trainer with optimized settings
    print("\n3. Setting up trainer...")
    trainer = ShapeRecognitionTrainerWithClassification(model, save_path='best_model_large.pth')

    # Better optimizer settings for larger model
    trainer.optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-3,  # Higher initial LR
        weight_decay=1e-4,  # Weight decay for regularization
        betas=(0.9, 0.999)
    )

    # More aggressive LR scheduling
    trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer,
        mode='min',
        factor=0.5,
        patience=4,
        min_lr=1e-6
    )

    print(f"   Device: {trainer.device}")
    print(f"   Optimizer: AdamW")
    print(f"   Initial LR: 2e-3")
    print(f"   Weight decay: 1e-4")

    # 4. Train
    print("\n4. Starting training...")
    print(f"   Expected time: ~{NUM_EPOCHS * 2} minutes")
    print("=" * 80)

    history = trainer.train(train_loader, val_loader, num_epochs=NUM_EPOCHS)

    # 5. Test inference
    print("\n5. Testing final model...")
    model.load_state_dict(torch.load('best_model_large.pth'))
    inference = ShapeInferenceWithClassification(model)

    # Test on various shapes
    from data_generation_with_types import ShapeGeneratorWithTypes

    test_cases = [
        ("3x3 tiny", 3, 3),
        ("5x5 small", 5, 5),
        ("10x10 medium", 10, 10),
        ("15x15 large", 15, 15),
        ("25x25 very large", 25, 25),
    ]

    print("\n   Testing on various grid sizes:")
    for desc, h, w in test_cases:
        generator = ShapeGeneratorWithTypes(NUM_COLORS, min_size=h, max_size=h)
        sample = generator.generate_sample()

        # Ensure correct size
        attempts = 0
        while (sample['grid'].shape[0] != h or sample['grid'].shape[1] != w) and attempts < 20:
            sample = generator.generate_sample()
            attempts += 1

        test_grid = sample['grid']
        prediction = inference.predict(test_grid)

        # Get ground truth for comparison
        gt_types = []
        if 'metadata' in sample:
            meta = sample['metadata']
            if 'type' in meta:
                gt_types = [meta['type']]
            elif 'shapes' in meta:
                gt_types = [s['type'] for s in meta['shapes']]

        pred_types = [obj['type'] for obj in prediction['objects']]

        print(f"\n   {desc}: GT={gt_types}, Pred={pred_types}")

    # 6. Final statistics
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    final_metrics = history['val_metrics'][-1] if history['val_metrics'] else {}

    print(f"\nFinal Validation Metrics:")
    print(f"  - Shape Classification Accuracy: {final_metrics.get('shape_class_accuracy', 0)*100:.1f}%")
    print(f"  - Instance Accuracy: {final_metrics.get('instance_accuracy', 0)*100:.1f}%")
    print(f"  - Vertex Precision: {final_metrics.get('vertex_precision', 0)*100:.1f}%")
    print(f"  - Vertex Recall: {final_metrics.get('vertex_recall', 0)*100:.1f}%")
    print(f"  - Edge IoU: {final_metrics.get('edge_iou', 0)*100:.1f}%")

    print(f"\nModel saved to: best_model_large.pth")
    print(f"Training history saved internally")

    print("\nNext steps:")
    print("  1. Run: python analyze_classification_performance.py 2000")
    print("  2. Run: python view_single_examples.py")
    print("  3. Compare with baseline performance")

    return model, history, inference


if __name__ == "__main__":
    model, history, inference = main()

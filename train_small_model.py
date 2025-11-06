"""
Training Script with Learned Shape Classification
Uses neural network to classify shapes instead of heuristics
"""

import torch
from torch.utils.data import DataLoader

from data_generation_with_types import ShapeDatasetWithTypes
from model_small import ShapeNetWithClassification
from training_with_classification import ShapeRecognitionTrainerWithClassification
from inference_with_classification import ShapeInferenceWithClassification
from utils import collate_variable_size


def main():
    """Main training function with learned classification."""

    # Configuration
    NUM_COLORS = 10
    MAX_INSTANCES = 10
    BATCH_SIZE = 32
    NUM_EPOCHS = 40
    TRAIN_SAMPLES = 100000
    VAL_SAMPLES = 5000

    print("=" * 80)
    print("SHAPE RECOGNITION WITH LEARNED CLASSIFICATION")
    print("=" * 80)
    print("\nKey improvements over heuristic approach:")
    print("  - Neural network learns to classify shape types")
    print("  - No brittle vertex-counting heuristics")
    print("  - Shape classification trained end-to-end")
    print("  - Multi-task learning with classification loss")
    print("=" * 80)

    # 1. Create datasets with shape type labels
    print("\n1. Creating/loading training data with type labels...")
    train_dataset = ShapeDatasetWithTypes(
        TRAIN_SAMPLES, NUM_COLORS, min_size=1, max_size=30,
        cache_path='data/train_dataset_small_100k.pkl'
    )
    val_dataset = ShapeDatasetWithTypes(
        VAL_SAMPLES, NUM_COLORS, min_size=1, max_size=30,
        cache_path='data/val_dataset_small_5k.pkl'
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_variable_size)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, collate_fn=collate_variable_size)

    print(f"   Training samples: {TRAIN_SAMPLES}")
    print(f"   Validation samples: {VAL_SAMPLES}")

    # 2. Create model with classification head
    print("\n2. Initializing model with shape classification head...")
    model = ShapeNetWithClassification(NUM_COLORS, MAX_INSTANCES)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Shape classes: {ShapeNetWithClassification.NUM_SHAPE_CLASSES}")
    print(f"   Classes: {list(ShapeNetWithClassification.SHAPE_CLASSES.keys())}")

    # 3. Train
    print("\n3. Training model with shape classification loss...")
    trainer = ShapeRecognitionTrainerWithClassification(model, save_path='best_model_small.pth')
    print(f"   Device: {trainer.device}")

    history = trainer.train(train_loader, val_loader, num_epochs=NUM_EPOCHS)

    # 4. Test inference
    print("\n4. Testing inference with learned classification...")
    model.load_state_dict(torch.load('best_model_small.pth'))
    inference = ShapeInferenceWithClassification(model)

    # Test on simple shapes
    from data_generation_with_types import ShapeGeneratorWithTypes

    simple_tests = [
        ("5x5 line", 5, 5),
        ("7x7 triangle", 7, 7),
        ("10x10 rectangle", 10, 10),
        ("12x12 circle", 12, 12),
        ("15x15 mixed", 15, 15),
    ]

    print("\n   Testing on various grid sizes:")
    for desc, h, w in simple_tests:
        generator = ShapeGeneratorWithTypes(NUM_COLORS, min_size=h, max_size=h)
        sample = generator.generate_sample()

        # Try to get the desired size
        attempts = 0
        while (sample['grid'].shape[0] != h or sample['grid'].shape[1] != w) and attempts < 10:
            sample = generator.generate_sample()
            attempts += 1

        test_grid = sample['grid']
        prediction = inference.predict(test_grid)

        print(f"\n   {desc}:")
        print(f"      Detected objects: {len(prediction['objects'])}")
        for i, obj in enumerate(prediction['objects']):
            print(f"        {i+1}. {obj['type']} (area: {obj['properties']['area']}, vertices: {obj['properties']['num_vertices']})")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - SMALL MODEL")
    print("=" * 80)
    print(f"\nBest model saved to: best_model_small.pth")
    print(f"Training data saved to: data/train_dataset_small_100k.pkl")
    print(f"Validation data saved to: data/val_dataset_small_5k.pkl")

    print("\nNext steps:")
    print("  1. Run: python analyze_classification_performance.py 1000 best_model_small.pth")
    print("  2. Run: python view_single_examples.py")
    print("  3. Compare with large model: python compare_models.py 500")

    return model, history, inference


if __name__ == "__main__":
    model, history, inference = main()

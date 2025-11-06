"""
Improved Training Script with Better Hyperparameters
Focuses on achieving high accuracy on simple shapes
"""

import torch
from torch.utils.data import DataLoader

from data_generation import ShapeDataset
from model import StructuralShapeNet
from training import ShapeRecognitionTrainer
from inference import ShapeInference
from utils import collate_variable_size


def main():
    """Main training function with improved hyperparameters."""

    # IMPROVED CONFIGURATION
    NUM_COLORS = 8
    MAX_INSTANCES = 10
    BATCH_SIZE = 32  # Increased from 16
    NUM_EPOCHS = 50  # Increased from 30
    TRAIN_SAMPLES = 10000  # Increased from 5000
    VAL_SAMPLES = 1000  # Increased from 500

    print("=" * 80)
    print("IMPROVED SHAPE RECOGNITION TRAINING")
    print("=" * 80)
    print("\nKey improvements:")
    print("  - Larger batch size (32 vs 16)")
    print("  - More epochs (50 vs 30)")
    print("  - More training data (10k vs 5k)")
    print("  - Better learning rate schedule")
    print("  - Focused on simple shapes")
    print("=" * 80)

    # 1. Create datasets with caching
    print("\n1. Creating/loading training data...")
    train_dataset = ShapeDataset(
        TRAIN_SAMPLES, NUM_COLORS, min_size=1, max_size=30,
        cache_path='data/train_dataset_large.pkl'
    )
    val_dataset = ShapeDataset(
        VAL_SAMPLES, NUM_COLORS, min_size=1, max_size=30,
        cache_path='data/val_dataset_large.pkl'
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_variable_size)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, collate_fn=collate_variable_size)

    print(f"   Training samples: {TRAIN_SAMPLES}")
    print(f"   Validation samples: {VAL_SAMPLES}")

    # 2. Create model
    print("\n2. Initializing model...")
    model = StructuralShapeNet(NUM_COLORS, MAX_INSTANCES)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # 3. Create trainer with improved settings
    print("\n3. Setting up trainer...")
    trainer = ShapeRecognitionTrainer(model)

    # Adjust learning rate and scheduler for better convergence
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)  # Higher initial LR
    trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, mode='min', factor=0.5, patience=3
    )

    print(f"   Device: {trainer.device}")
    print(f"   Initial learning rate: 2e-3")

    # 4. Train
    print("\n4. Training model...")
    history = trainer.train(train_loader, val_loader, num_epochs=NUM_EPOCHS)

    # 5. Test inference on simple shapes
    print("\n5. Testing on simple shapes...")
    model.load_state_dict(torch.load('best_model.pth'))
    inference = ShapeInference(model)

    # Test on simple shapes specifically
    from data_generation import ShapeGenerator

    simple_tests = [
        ("3x3 rectangle", 3, 3),
        ("5x5 rectangle", 5, 5),
        ("7x7 rectangle", 7, 7),
        ("10x10 mixed", 10, 10),
    ]

    for desc, h, w in simple_tests:
        generator = ShapeGenerator(NUM_COLORS, min_size=h, max_size=h)
        sample = generator.generate_sample()
        while sample['grid'].shape[0] != h or sample['grid'].shape[1] != w:
            sample = generator.generate_sample()

        test_grid = sample['grid']
        prediction = inference.predict(test_grid)

        print(f"\n   {desc}:")
        print(f"      Detected objects: {len(prediction['objects'])}")
        for i, obj in enumerate(prediction['objects']):
            print(f"        {i+1}. {obj['type']} (vertices: {obj['properties']['num_vertices']})")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest model saved to: best_model.pth")
    print(f"Training data saved to: data/train_dataset_large.pkl")
    print(f"Validation data saved to: data/val_dataset_large.pkl")

    print("\nNext steps:")
    print("  1. Run: python analyze_performance.py")
    print("  2. Run: python view_training_samples.py")
    print("  3. Run: python view_single_examples.py")

    return model, history, inference


if __name__ == "__main__":
    model, history, inference = main()

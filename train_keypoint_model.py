"""
Training Script for Keypoint Regression Model
Trains a model that directly predicts vertex coordinates
"""

import torch
from torch.utils.data import DataLoader
from model_with_keypoints import ShapeNetWithKeypoints
from data_generation_with_keypoints import ShapeDatasetWithKeypoints
from training_with_keypoints import KeypointTrainer
from utils import collate_keypoint_dataset


def train_keypoint_model():
    """Train the keypoint regression model."""

    # Hyperparameters
    NUM_COLORS = 10
    MAX_INSTANCES = 10
    TRAIN_SAMPLES = 100000  # 5000 # Start smaller for testing
    VAL_SAMPLES = 5000
    BATCH_SIZE = 32
    NUM_EPOCHS = 60
    EARLY_STOPPING_PATIENCE = 12

    print("="*80)
    print("KEYPOINT REGRESSION MODEL TRAINING")
    print("="*80)
    print(f"Training samples: {TRAIN_SAMPLES}")
    print(f"Validation samples: {VAL_SAMPLES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max epochs: {NUM_EPOCHS}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print("="*80)

    # Create datasets
    print("\nPreparing datasets...")
    train_dataset = ShapeDatasetWithKeypoints(
        num_samples=TRAIN_SAMPLES,
        num_colors=NUM_COLORS,
        min_size=5,
        max_size=25,
        max_instances=MAX_INSTANCES,
        cache_path='data/train_keypoints_100k.pkl'
    )

    val_dataset = ShapeDatasetWithKeypoints(
        num_samples=VAL_SAMPLES,
        num_colors=NUM_COLORS,
        min_size=5,
        max_size=25,
        max_instances=MAX_INSTANCES,
        cache_path='data/val_keypoints_5000.pkl'
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_keypoint_dataset,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_keypoint_dataset,
        num_workers=0
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("\nInitializing model...")
    model = ShapeNetWithKeypoints(num_colors=NUM_COLORS, max_instances=MAX_INSTANCES)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Create trainer
    trainer = KeypointTrainer(
        model,
        save_path='best_model_keypoints.pth'
    )

    print(f"\nDevice: {trainer.device}")

    # Train
    print("\n" + "="*80)
    print("TRAINING START")
    print("="*80 + "\n")

    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best model saved to: best_model_keypoints.pth")

    return history


if __name__ == "__main__":
    history = train_keypoint_model()

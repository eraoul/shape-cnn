"""
Main Entry Point for Shape Recognition Pipeline
Coordinates training and inference of the complete pipeline
"""

import torch
from torch.utils.data import DataLoader

from data_generation import ShapeGenerator, ShapeDataset
from model import StructuralShapeNet
from training import ShapeRecognitionTrainer
from inference import ShapeInference
from utils import collate_variable_size


def main():
    """Main execution function."""

    # Configuration
    NUM_COLORS = 10
    MAX_INSTANCES = 10
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    TRAIN_SAMPLES = 5000
    VAL_SAMPLES = 500

    print("=" * 80)
    print("SHAPE RECOGNITION TRAINING PIPELINE")
    print("=" * 80)

    # 1. Create datasets with caching
    print("\n1. Creating/loading training data...")
    train_dataset = ShapeDataset(
        TRAIN_SAMPLES, NUM_COLORS, min_size=1, max_size=30,
        cache_path='data/train_dataset.pkl'
    )
    val_dataset = ShapeDataset(
        VAL_SAMPLES, NUM_COLORS, min_size=1, max_size=30,
        cache_path='data/val_dataset.pkl'
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

    # 3. Train
    print("\n3. Training model...")
    trainer = ShapeRecognitionTrainer(model)
    history = trainer.train(train_loader, val_loader, num_epochs=NUM_EPOCHS)

    # 4. Test inference
    print("\n4. Testing inference...")
    model.load_state_dict(torch.load('best_model.pth'))
    inference = ShapeInference(model)

    # Generate a test sample
    generator = ShapeGenerator(NUM_COLORS, min_size=10, max_size=10)
    test_sample = generator.generate_sample()
    test_grid = test_sample['grid']

    # Run inference
    prediction = inference.predict(test_grid)

    print(f"\n   Test grid size: {prediction['grid_size']}")
    print(f"   Detected objects: {len(prediction['objects'])}")

    for i, obj in enumerate(prediction['objects']):
        print(f"\n   Object {i+1}:")
        print(f"      Type: {obj['type']}")
        print(f"      Color: {obj['color']}")
        print(f"      Vertices: {len(obj['vertices'])}")
        print(f"      Area: {obj['properties']['area']} cells")
        print(f"      Filled: {obj['properties']['is_filled']}")

    # Visualize
    print("\n5. Generating visualization...")
    inference.visualize(test_grid, prediction, save_path='inference_result.png')

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest model saved to: best_model.pth")
    print(f"Visualization saved to: inference_result.png")

    return model, history, inference


if __name__ == "__main__":
    model, history, inference = main()

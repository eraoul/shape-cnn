"""
Complete ML Pipeline for Grid-Based Shape Recognition
Handles 1x1 to 30x30+ grids with 8 discrete colors
Recognizes lines, triangles, rectangles, circles, and irregular shapes
Outputs structural information (vertices, edges, perimeters)

REFACTORED: This file now serves as a compatibility layer.
The pipeline has been split into separate modules for better organization:

- data_generation.py: ShapeGenerator and ShapeDataset classes
- model.py: StructuralShapeNet neural network architecture
- training.py: ShapeRecognitionTrainer for training loop
- inference.py: ShapeInference for prediction and visualization
- utils.py: Helper functions including collate_variable_size
- main.py: Main entry point for training and inference

Usage:
    # For backward compatibility, you can still run this file:
    python shape_recognition_pipeline.py

    # Or use the new main.py entry point:
    python main.py

    # Or import individual components:
    from data_generation import ShapeGenerator, ShapeDataset
    from model import StructuralShapeNet
    from training import ShapeRecognitionTrainer
    from inference import ShapeInference
    from utils import collate_variable_size
"""

# Import all components from refactored modules
from data_generation import ShapeGenerator, ShapeDataset
from model import StructuralShapeNet
from training import ShapeRecognitionTrainer
from inference import ShapeInference
from utils import collate_variable_size
from main import main

# Maintain backward compatibility
__all__ = [
    'ShapeGenerator',
    'ShapeDataset',
    'StructuralShapeNet',
    'ShapeRecognitionTrainer',
    'ShapeInference',
    'collate_variable_size',
    'main'
]


if __name__ == "__main__":
    # Run the main training pipeline for backward compatibility
    model, history, inference = main()

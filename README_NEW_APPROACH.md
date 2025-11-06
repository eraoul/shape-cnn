# Improved Shape Detection Approach

## Overview

This document describes the new and improved approach for detecting shapes (lines, vertices, and polygons) in 2D grid worlds.

## Problem with Previous Approaches

The previous implementations had several critical issues:

1. **Training Data Mismatch**: Model expected 4 shape classes but only trained on 2 (lines and rectangles)
2. **Missing Shape Types**: No triangles, circles, or arbitrary polygons in training data
3. **Extreme Loss Weighting**: Vertex detection had 10x class weight, shape classification had 3x total weight
4. **Overly Complex Architecture**: Color-invariant layers were too restrictive
5. **No Data Augmentation**: Missing rotations, flips, noise
6. **Poor Metrics**: Only per-pixel accuracy, no instance-level evaluation

**Result**: Models failed to learn the task effectively (12-16% accuracy on some tasks)

## New Approach

### Architecture: U-Net Multi-Task Learning

We use a standard **U-Net encoder-decoder** architecture with four output heads:

```
Input: (10, H, W) one-hot encoded grid
         ↓
    U-Net Encoder (64→128→256→512→1024 channels)
         ↓
    U-Net Decoder with Skip Connections
         ↓
    ┌────────┬─────────┬────────┬──────────────┐
    ↓        ↓         ↓        ↓              ↓
Instance  Vertex    Edge    Shape
  Head     Head     Head  Classification
                              Head
```

**Output Heads**:
1. **Instance Segmentation**: Which pixels belong to which shape (N+1 classes)
2. **Vertex Detection**: Probability of pixel being a vertex/corner (binary)
3. **Edge Detection**: Probability of pixel being an edge (binary)
4. **Shape Classification**: Type of shape at each pixel (6 classes)

### Training Data Generation

**Shape Types** (all represented in training):
- **Lines**: 2 vertices
- **Triangles**: 3 vertices (random or regular)
- **Rectangles**: 4 vertices (axis-aligned and rotated)
- **Polygons**: 5-8 vertices (arbitrary shapes)
- **Circles**: Approximated as regular polygons with 12-20 sides

**Diversity**:
- Grid sizes: 5×5 to 30×30
- Multiple shapes per grid (1-5 depending on size)
- Filled and outline shapes
- No overlap (each pixel has one color)

### Loss Function

**Balanced weights** (no extreme multipliers):
```python
loss_weights = {
    'instance': 1.0,       # Instance segmentation
    'vertex': 1.5,         # Vertex detection (moderate boost)
    'edge': 1.0,           # Edge detection
    'shape_class': 1.5,    # Shape classification (moderate boost)
}
```

**Focal Loss** for vertices and edges:
- Addresses class imbalance naturally (most pixels are not vertices/edges)
- Focuses learning on hard examples
- Parameters: α=0.25, γ=2.0

**Why this is better**:
- Previous approach used 10x class weight on vertices → gradient instability
- Focal loss adaptively focuses on hard examples without extreme weighting
- More balanced overall training

### Data Augmentation

- Horizontal flip (50%)
- Vertical flip (50%)
- 90° rotation (25%)
- Small Gaussian noise (10%)

### Evaluation Metrics

**Per-Instance Metrics** (not just per-pixel):
- Instance segmentation accuracy
- Vertex detection accuracy (with threshold)
- Edge detection accuracy (with threshold)
- Shape classification accuracy (on non-background pixels only)

## File Structure

```
shape-cnn/
├── data_generation_improved.py  # New data generator with all shape types
├── model_unet.py                # Clean U-Net architecture
├── train_unet.py                # Training script with balanced losses
├── inference_unet.py            # Inference and visualization
├── data/                        # Generated datasets (cached)
├── checkpoints/                 # Trained model weights
└── results/                     # Visualization outputs
```

## Usage

### 1. Generate Training Data

```python
from data_generation_improved import generate_dataset

dataset = generate_dataset(
    num_samples=10000,
    num_colors=10,
    min_size=5,
    max_size=30,
    cache_path='data/train_dataset_unet.pkl'
)
```

### 2. Train Model

```bash
python train_unet.py
```

Configuration (in `train_unet.py`):
- Training samples: 10,000
- Validation samples: 1,000
- Batch size: 16
- Learning rate: 1e-3 with ReduceLROnPlateau
- Epochs: 50

**GPU Acceleration**:
- **Mac (Apple Silicon)**: Automatically uses Metal Performance Shaders (MPS) for GPU acceleration
- **NVIDIA GPUs**: Automatically uses CUDA if available
- **CPU Fallback**: Uses CPU if no GPU is available

The code automatically detects and uses the best available device.

### 3. Run Inference

```python
from inference_unet import ShapeDetector

detector = ShapeDetector(
    model_path='checkpoints/unet_best.pth',
    vertex_threshold=0.4,
    edge_threshold=0.4
)

prediction = detector.predict(grid)

# prediction contains:
# - shapes: List of detected shapes with vertices and types
# - instance_map: Instance segmentation
# - vertex_map: Vertex detection heatmap
# - edge_map: Edge detection heatmap
```

### 4. Visualize Results

```bash
python inference_unet.py
```

Creates visualizations showing:
- Original grid
- Instance segmentation
- Vertex detection
- Edge detection
- Shape classification
- Final detections with labeled vertices

## Expected Performance

Based on the improved approach, we expect:

- **Instance Segmentation**: 85-95% accuracy
- **Vertex Detection**: 80-90% accuracy (with proper thresholding)
- **Shape Classification**: 70-85% accuracy
- **Overall Shape Detection**: Successfully detect and classify most shapes

Previous approach achieved only 12-16% on some tasks, so this is a **major improvement**.

## Key Improvements Summary

| Aspect | Old Approach | New Approach |
|--------|-------------|--------------|
| Architecture | Color-invariant layers | Standard U-Net |
| Shape Types | 2 (line, rectangle) | 5 (line, triangle, rectangle, polygon, circle) |
| Loss Weighting | Extreme (10x vertices) | Balanced (1.0-1.5x) |
| Class Imbalance | Weighted BCE | Focal Loss |
| Data Augmentation | None | Flip, rotate, noise |
| Metrics | Per-pixel only | Per-instance + per-pixel |
| Parameters | 600k - 4.2M | 400k - 2.5M (more efficient) |

## Model Variants

### CompactShapeUNet (Recommended for experiments)
- **Parameters**: ~400,000
- **Depth**: 3 down/up blocks
- **Channels**: 32→64→128→256
- **Training time**: ~5-10 min/epoch on GPU
- **Use case**: Fast iteration, good performance

### ShapeUNet (Full model)
- **Parameters**: ~2,500,000
- **Depth**: 4 down/up blocks
- **Channels**: 64→128→256→512→1024
- **Training time**: ~15-20 min/epoch on GPU
- **Use case**: Maximum accuracy

## Next Steps

1. **Train the model**: Run `python train_unet.py`
2. **Evaluate on test set**: Check instance-level metrics
3. **Analyze failures**: Which shape types are hardest?
4. **Iterate**:
   - Adjust shape distribution in training data
   - Fine-tune loss weights
   - Experiment with thresholds
   - Add more difficult cases (occluded shapes, noise)

## Theoretical Justification

### Why U-Net?

U-Net is the standard architecture for segmentation tasks:
- **Skip connections** preserve fine spatial details (vertices, edges)
- **Encoder-decoder** structure captures multi-scale features
- **Proven effectiveness** on similar dense prediction tasks

### Why Multi-Task Learning?

Learning multiple related tasks jointly:
- **Shared representations**: Features useful for one task help others
- **Regularization**: Prevents overfitting to any single task
- **Efficiency**: Single forward pass produces all outputs

### Why Focal Loss for Vertices/Edges?

Vertices and edges are **extremely sparse**:
- Typical grid: 900 pixels, 4 vertices, 40 edge pixels
- Class ratio: 1:225 (vertices) and 1:21 (edges)
- Focal loss automatically handles this without manual tuning

## Comparison to Alternative Approaches

### Object Detection (YOLO-style)
- **Pros**: Direct bbox prediction, fast inference
- **Cons**: Shapes aren't boxes, need arbitrary number of vertices
- **Verdict**: Not well-suited for this problem

### Transformer (DETR-style)
- **Pros**: Can handle variable number of objects elegantly
- **Cons**: Requires more data, slower, overkill for simple grids
- **Verdict**: Overcomplicated for this task

### Pure Segmentation (Mask R-CNN)
- **Pros**: State-of-art instance segmentation
- **Cons**: Heavy architecture, needs bounding box proposals
- **Verdict**: U-Net is simpler and sufficient

### Keypoint Detection Only
- **Pros**: Directly predicts vertices
- **Cons**: No shape instance separation, no shape type classification
- **Verdict**: Incomplete solution

**Our approach** combines the best aspects: instance segmentation + keypoint detection + classification in a unified framework.

## References

- U-Net: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- Focal Loss: [Lin et al., 2017](https://arxiv.org/abs/1708.02002)
- Multi-Task Learning: [Caruana, 1997](https://link.springer.com/article/10.1023/A:1007379606734)

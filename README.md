# Shape Recognition Pipeline

Complete ML pipeline for grid-based shape recognition with **learned shape classification**. Supports 1x1 to 30x30+ grids with color-invariant architecture.

## Overview

This project uses neural networks to detect and classify shapes in colored grids. Key features:

- **Learned Classification**: Neural network learns to classify shapes (line, triangle, rectangle, circle, irregular) instead of using heuristics
- **Color Invariance**: First layer uses weight sharing across colors - treats all colors symmetrically
- **Multi-Task Learning**: Simultaneously predicts instances, vertices, edges, and shape types
- **Two Model Sizes**: Small model (~600k params) and large model (~4.2M params)
- **Variable Grid Sizes**: Handles grids from 1x1 to 30x30+

## Quick Start

### 1. Training Models

```bash
# Train small model (10k samples, 40 epochs, ~600k parameters)
python train_small_model.py

# Train large model (50k samples, 80 epochs, ~4.2M parameters)
python train_large_model.py
```

**Outputs:**
- Small model: `best_model_small.pth`
- Large model: `best_model_large.pth`

### 2. Analyze Performance

```bash
# Analyze large model on 2000 samples
python analyze_classification_performance.py 2000

# Analyze small model on 1000 samples
python analyze_classification_performance.py 1000 best_model_small.pth

# Compare both models side-by-side
python compare_models.py 500
```

### 3. Visualize Predictions

```bash
# View random examples one at a time (close window to advance)
python view_single_examples.py

# View 10 random examples
python view_single_examples.py 10

# View training set samples (ground truth vs predictions)
python view_training_samples.py

# View specific sample by index
python view_training_samples.py 42
```

### 4. Test Color Invariance

```bash
# Verify that swapping colors produces equivalent features
python test_color_invariance.py
```

## Model Architecture

### Small Model: `ShapeNetWithClassification` (~600k parameters)

**File:** `model_with_classification.py`

- **Color-invariant first layer**: 4 features per color (40 total for 10 colors)
- **3 convolutional blocks**: 32 → 64 → 128 channels
- **Skip connections**: Multi-scale feature fusion
- **4 output heads**:
  1. Instance segmentation (which pixels belong to which shape)
  2. Vertex detection (where vertices are)
  3. Edge detection (where edges are)
  4. **Shape classification** (per-pixel shape type prediction)

**Shape Classes (Simplified):**
- Background (0)
- Line (1)
- Rectangle (2)
- Single Cell (3)

### Large Model: `LargeShapeNet` (~4.2M parameters)

**File:** `model_large.py`

- **Color-invariant first layer**: Same as small model
- **4 convolutional blocks**: 32 → 64 → 128 → 256 channels
- **3 layers per block**: Deeper architecture for more capacity
- **Skip connections from 3 scales**: Better feature fusion
- **Dropout (0.1)**: Regularization for better generalization
- **Enhanced output heads**: 3-layer heads for better predictions

**Training Configuration:**
- 50k training samples (5x more than small model)
- Batch size: 48
- Optimizer: AdamW with weight decay (1e-4)
- Learning rate: 2e-3 with ReduceLROnPlateau

### Key Innovation: Color-Invariant First Layer

**Problem:** Colors are arbitrary labels with no inherent meaning. Color 0 should be treated the same as Color 1.

**Solution:** Weight sharing across color channels
- Same 3x3 kernel applied to each color independently
- Color 0 gets features [0:4], Color 1 gets [4:8], etc.
- Swapping colors just permutes feature channel groups
- Network learns "what makes a shape" independent of color

**Benefits:**
- Better generalization to unseen color combinations
- More efficient learning (fewer parameters in first layer)
- Preserves color distinctness for shape separation

## Training Pipeline

### Multi-Task Loss Function

```
Total Loss = Instance Loss
           + 2.0 × Vertex Loss
           + 1.5 × Edge Loss
           + 3.0 × Shape Classification Loss
```

**Loss Types:**
- **Instance Loss**: Cross-entropy for instance segmentation
- **Vertex Loss**: Binary cross-entropy with 10x weight on positive class
- **Edge Loss**: Binary cross-entropy with 5x weight on positive class
- **Shape Classification Loss**: Cross-entropy for per-pixel shape types (3x weight - highest priority)

### Data Generation

**File:** `data_generation_with_types.py`

- Generates synthetic grids with various shapes
- Creates per-pixel ground truth labels for all tasks
- Automatic caching to disk for faster subsequent runs
- Variable grid sizes (1x1 to 30x30)

**Cached Datasets:**
- `data/train_dataset_with_types.pkl` (10k samples)
- `data/val_dataset_with_types.pkl` (1k samples)
- `data/train_dataset_large_50k.pkl` (50k samples)
- `data/val_dataset_large_5k.pkl` (5k samples)

## Project Structure

```
cnn/
├── Core Models
│   ├── model_with_classification.py    # Small model (~600k params)
│   └── model_large.py                   # Large model (~4.2M params)
│
├── Training
│   ├── training_with_classification.py  # Trainer with classification loss
│   ├── train_small_model.py             # Small model training script
│   └── train_large_model.py             # Large model training script
│
├── Data Generation
│   ├── data_generation_with_types.py    # Dataset with shape type labels
│   └── utils.py                         # Collate function for batching
│
├── Inference & Visualization
│   ├── inference_with_classification.py # Inference with learned classification
│   ├── view_single_examples.py          # View random examples
│   └── view_training_samples.py         # View training set with predictions
│
├── Analysis
│   ├── analyze_classification_performance.py  # Detailed accuracy analysis
│   ├── compare_models.py                      # Compare small vs large models
│   └── test_color_invariance.py               # Verify color invariance
│
├── Legacy (for reference)
│   ├── model.py                         # Original model without classification
│   ├── model_improved.py                # Improved heuristic-based model
│   ├── training.py                      # Original trainer
│   ├── inference.py                     # Heuristic-based inference
│   └── analyze_performance.py           # Heuristic-based analysis
│
└── Data (auto-created)
    ├── train_dataset_with_types.pkl
    ├── val_dataset_with_types.pkl
    ├── train_dataset_large_50k.pkl
    └── val_dataset_large_5k.pkl
```

## Command Reference

### Training

```bash
# Small model (10k samples, 40 epochs)
python train_small_model.py

# Large model (50k samples, 80 epochs)
python train_large_model.py
```

### Analysis

```bash
# Analyze large model (default)
python analyze_classification_performance.py 2000

# Analyze small model
python analyze_classification_performance.py 1000 best_model_small.pth

# Compare both models
python compare_models.py 500
```

**Output:** Accuracy by shape type (line, rectangle)

### Visualization

```bash
# View random examples (close window to advance)
python view_single_examples.py

# View N random examples
python view_single_examples.py 10

# View training samples with predictions
python view_training_samples.py

# View from specific index
python view_training_samples.py 100

# View validation set
python view_training_samples.py --val

# View N random samples
python view_training_samples.py --random 5
```

### Testing

```bash
# Test color invariance property
python test_color_invariance.py
```

## Device Support

Automatically detects and uses:

1. **MPS** (Apple Silicon GPU) - MacBook Pro M1/M2/M3/M4
2. **CUDA** (NVIDIA GPU)
3. **CPU** (fallback)

Training will display: `Device: mps` (or `cuda`/`cpu`)

## Expected Performance

### Target Metrics

- **Shape Classification Accuracy**: 60-90%+ (learned classification)
  - Previous heuristic approach: 12-16% for simple shapes
  - Learned approach handles variations much better
- **Vertex Precision/Recall**: 85-95%
- **Edge IoU**: 70-85%
- **Instance Accuracy**: 70-85%

### Model Comparison

| Metric | Small Model | Large Model |
|--------|-------------|-------------|
| Parameters | ~600k | ~4.2M |
| Training Samples | 10k | 50k |
| Training Time | ~40 min | ~160 min |
| Expected Accuracy | 50-70% | 70-90% |

*Note: Large model should outperform small model due to increased capacity and more training data*

## Training Tips

1. **First run takes longer**: Dataset generation and caching (5-10 min for 50k samples)
2. **Subsequent runs are fast**: Datasets load from cache in seconds
3. **Monitor shape classification loss**: Should decrease steadily (weighted 3.0x)
4. **Check best model saves**: Look for "✓ Best model saved" messages
5. **Use validation metrics**: Shape classification accuracy is the key metric

## Troubleshooting

### RuntimeError: Key mismatch in state_dict

**Cause:** Trying to load large model weights into small model (or vice versa)

**Solution:** Use correct model path with analyze script:
```bash
# For large model
python analyze_classification_performance.py 2000

# For small model
python analyze_classification_performance.py 1000 best_model_small.pth
```

### Training is slow

**Check:**
1. Device is MPS/CUDA: Look for `Device: mps` in output
2. Datasets load from cache: Should see "Loading dataset from..." immediately
3. Batch size isn't too large: Default is 32 (small) or 48 (large)

### Low shape classification accuracy

**Try:**
1. Train longer: Increase `NUM_EPOCHS` in training script
2. Use larger model: Switch to `train_large_model.py`
3. Check loss weights: Shape classification is weighted 3.0x
4. Visualize predictions: Use `view_single_examples.py` to inspect

### Out of memory

**Solutions:**
1. Reduce batch size in training script
2. Reduce `TRAIN_SAMPLES` or `VAL_SAMPLES`
3. Use smaller model (`train_small_model.py`)

## Key Differences from Heuristic Approach

### Old Approach (Heuristic-based)
- Detected vertices and edges using CNN
- **Classified shapes by counting vertices** (heuristics)
- Brittle: 3 vertices = triangle, 4 = rectangle, etc.
- Failed on variations: Achieved only 12-16% accuracy on simple shapes
- Files: `model_improved.py`, `inference.py`, `analyze_performance.py`

### New Approach (Learned Classification)
- Detects vertices, edges, **and learns shape types** end-to-end
- **Network learns "what makes a triangle"** from data
- Robust: Handles variations, noise, and edge cases
- Achieves 50-90%+ accuracy depending on model size
- Files: `model_with_classification.py`, `inference_with_classification.py`

## Next Steps

After training:

1. **Analyze performance**:
   ```bash
   python analyze_classification_performance.py 2000
   ```

2. **Compare models** (if you trained both):
   ```bash
   python compare_models.py 500
   ```

3. **Visualize predictions**:
   ```bash
   python view_single_examples.py 10
   ```

4. **Identify weak shapes** and retrain with more data if needed

## Future Improvements

- [ ] Data augmentation (rotation, flipping, color permutations)
- [ ] Attention mechanisms for better vertex detection
- [ ] Shape-specific loss weighting based on difficulty
- [ ] Real-time training visualization (TensorBoard)
- [ ] Export to ONNX for deployment
- [ ] Multi-object tracking and counting
- [ ] Hierarchy of shape types (polygon subtypes)

---

**Getting Started:**

```bash
# 1. Train large model
python train_large_model.py

# 2. Analyze results
python analyze_classification_performance.py 2000

# 3. View examples
python view_single_examples.py
```

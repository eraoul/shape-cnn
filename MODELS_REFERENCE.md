# Model Reference Card

Quick reference for the two model configurations.

## File Naming Convention

| Component | Small Model | Large Model |
|-----------|-------------|-------------|
| **Training Script** | `train_small_model.py` | `train_large_model.py` |
| **Model Class File** | `model_with_classification.py` | `model_large.py` |
| **Model Class Name** | `ShapeNetWithClassification` | `LargeShapeNet` |
| **Saved Checkpoint** | `best_model_small.pth` | `best_model_large.pth` |
| **Training Dataset** | `data/train_dataset_with_types.pkl` | `data/train_dataset_large_50k.pkl` |
| **Validation Dataset** | `data/val_dataset_with_types.pkl` | `data/val_dataset_large_5k.pkl` |

## Model Specifications

### Small Model

```python
# Architecture
- File: model_with_classification.py
- Class: ShapeNetWithClassification
- Parameters: ~600k
- Conv Blocks: 3 (32→64→128)
- Layers per Block: 2
- Skip Connections: 2
- Dropout: None

# Training
- Script: train_small_model.py
- Samples: 10,000 training + 1,000 validation
- Batch Size: 32
- Epochs: 40
- Optimizer: Adam (lr=1e-3)
- Expected Time: ~40 minutes
- Output: best_model_small.pth

# Usage
python train_small_model.py
python analyze_classification_performance.py 1000 best_model_small.pth
```

### Large Model

```python
# Architecture
- File: model_large.py
- Class: LargeShapeNet
- Parameters: ~4.2M
- Conv Blocks: 4 (32→64→128→256)
- Layers per Block: 3
- Skip Connections: 3
- Dropout: 0.1

# Training
- Script: train_large_model.py
- Samples: 50,000 training + 5,000 validation
- Batch Size: 48
- Epochs: 80
- Optimizer: AdamW (lr=2e-3, weight_decay=1e-4)
- Expected Time: ~160 minutes
- Output: best_model_large.pth

# Usage
python train_large_model.py
python analyze_classification_performance.py 2000
```

## Common Configuration (Both Models)

```python
# Data
- Colors: 10 (color 0 reserved for background)
- Max Instances: 10
- Grid Sizes: 1x1 to 30x30
- Shape Classes: 4 (background, line, rectangle, single_cell)

# Architecture
- Color-invariant first layer (weight sharing)
- Multi-task learning (4 output heads)
- Loss weights: instance=1.0, vertex=2.0, edge=1.5, shape_class=3.0

# Device Support
- MPS (Apple Silicon)
- CUDA (NVIDIA GPU)
- CPU (fallback)
```

## Quick Commands

### Training
```bash
# Small model (fast, good for testing)
python train_small_model.py

# Large model (slow, best accuracy)
python train_large_model.py
```

### Analysis
```bash
# Analyze small model
python analyze_classification_performance.py 1000 best_model_small.pth

# Analyze large model (default)
python analyze_classification_performance.py 2000

# Compare both models
python compare_models.py 500
```

### Visualization
```bash
# View examples (uses large model by default)
python view_single_examples.py

# View with specific model
# (modify view_single_examples.py line 27 to change model_path)
```

### Testing
```bash
# Test data generation
python test_simple_generation.py

# Test color invariance
python test_color_invariance.py
```

## When to Use Which Model

### Use Small Model When:
- ✓ Quick experimentation and iteration
- ✓ Testing code changes
- ✓ Limited compute resources
- ✓ Debugging the pipeline
- ✓ Initial validation of approach

### Use Large Model When:
- ✓ Training final production model
- ✓ Maximum accuracy needed
- ✓ Have 2+ hours for training
- ✓ Good GPU/MPS acceleration available
- ✓ Planning to add more complex shapes later

## Expected Performance (Simplified Shapes)

| Metric | Small Model | Large Model |
|--------|-------------|-------------|
| Line Accuracy | 85-90% | 90-95% |
| Rectangle Accuracy | 85-90% | 90-95% |
| Overall Accuracy | 85-90% | 90-95% |
| Training Time | 40 min | 160 min |
| Memory Usage | ~2 GB | ~4 GB |

## Troubleshooting

**"RuntimeError: Error(s) in loading state_dict"**
- You're trying to load wrong model type
- Check model path matches training script used
- Small model needs `best_model_small.pth`
- Large model needs `best_model_large.pth`

**"FileNotFoundError: best_model_X.pth"**
- Model hasn't been trained yet
- Run `train_small_model.py` or `train_large_model.py` first

**"Out of memory"**
- Use small model instead of large
- Reduce `BATCH_SIZE` in training script
- Reduce `TRAIN_SAMPLES` and `VAL_SAMPLES`

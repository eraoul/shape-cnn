"""
Training script for U-Net shape detection model.

Key improvements:
- Balanced loss weighting (no extreme 10x multipliers)
- Focal loss for vertices/edges to handle class imbalance
- Data augmentation (rotation, flip, noise)
- Proper evaluation metrics (per-instance accuracy)
- Learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List
import os
import pickle
from tqdm import tqdm

from model_unet import ShapeUNet, CompactShapeUNet
from data_generation_improved import generate_dataset, NUM_SHAPE_CLASSES


class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance.
    Reduces loss for well-classified examples, focuses on hard examples.
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        Args:
            pred: predictions in [0, 1] (after sigmoid)
            target: ground truth in {0, 1}
        """
        bce = F.binary_cross_entropy(pred, target, reduction='none')

        # Focal weight: (1 - p_t)^gamma
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha balancing
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        loss = alpha_t * focal_weight * bce
        return loss.mean()


class ShapeDataset(Dataset):
    """Dataset wrapper with augmentation."""

    def __init__(
        self,
        samples: List[Dict],
        augment: bool = True,
        max_instances: int = 10
    ):
        self.samples = samples
        self.augment = augment
        self.max_instances = max_instances

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx].copy()

        # Apply augmentation
        if self.augment:
            sample = self._augment(sample)

        # Convert to tensors
        grid = torch.from_numpy(sample['grid']).float().permute(2, 0, 1)
        instance_map = torch.from_numpy(sample['instance_map']).long()
        vertex_map = torch.from_numpy(sample['vertex_map']).float().unsqueeze(0)
        edge_map = torch.from_numpy(sample['edge_map']).float().unsqueeze(0)
        shape_type_map = torch.from_numpy(sample['shape_type_map']).long()

        return {
            'grid': grid,
            'instance_map': instance_map,
            'vertex_map': vertex_map,
            'edge_map': edge_map,
            'shape_type_map': shape_type_map,
        }

    def _augment(self, sample):
        """Apply random augmentation."""
        # 50% chance of horizontal flip
        if np.random.rand() < 0.5:
            sample['grid'] = np.fliplr(sample['grid']).copy()
            sample['instance_map'] = np.fliplr(sample['instance_map']).copy()
            sample['vertex_map'] = np.fliplr(sample['vertex_map']).copy()
            sample['edge_map'] = np.fliplr(sample['edge_map']).copy()
            sample['shape_type_map'] = np.fliplr(sample['shape_type_map']).copy()

        # 50% chance of vertical flip
        if np.random.rand() < 0.5:
            sample['grid'] = np.flipud(sample['grid']).copy()
            sample['instance_map'] = np.flipud(sample['instance_map']).copy()
            sample['vertex_map'] = np.flipud(sample['vertex_map']).copy()
            sample['edge_map'] = np.flipud(sample['edge_map']).copy()
            sample['shape_type_map'] = np.flipud(sample['shape_type_map']).copy()

        # 25% chance of 90-degree rotation
        if np.random.rand() < 0.25:
            k = np.random.randint(1, 4)  # 90, 180, or 270 degrees
            sample['grid'] = np.rot90(sample['grid'], k).copy()
            sample['instance_map'] = np.rot90(sample['instance_map'], k).copy()
            sample['vertex_map'] = np.rot90(sample['vertex_map'], k).copy()
            sample['edge_map'] = np.rot90(sample['edge_map'], k).copy()
            sample['shape_type_map'] = np.rot90(sample['shape_type_map'], k).copy()

        # Small amount of noise (10% chance)
        if np.random.rand() < 0.1:
            noise = np.random.randn(*sample['grid'].shape) * 0.05
            sample['grid'] = np.clip(sample['grid'] + noise, 0, 1).astype(np.float32)

        return sample


def collate_variable_size(batch):
    """Collate function that pads to max size in batch."""
    # Find max dimensions
    max_h = max(item['grid'].shape[1] for item in batch)
    max_w = max(item['grid'].shape[2] for item in batch)

    # Pad each item
    padded_grids = []
    padded_instances = []
    padded_vertices = []
    padded_edges = []
    padded_shape_types = []
    original_sizes = []

    for item in batch:
        _, h, w = item['grid'].shape
        original_sizes.append((h, w))

        # Pad each tensor
        pad_h = max_h - h
        pad_w = max_w - w

        grid = F.pad(item['grid'], (0, pad_w, 0, pad_h), value=0)
        grid[0, h:, :] = 1.0  # Background color
        grid[0, :, w:] = 1.0

        instance = F.pad(item['instance_map'], (0, pad_w, 0, pad_h), value=0)
        vertex = F.pad(item['vertex_map'], (0, pad_w, 0, pad_h), value=0)
        edge = F.pad(item['edge_map'], (0, pad_w, 0, pad_h), value=0)
        shape_type = F.pad(item['shape_type_map'], (0, pad_w, 0, pad_h), value=0)

        padded_grids.append(grid)
        padded_instances.append(instance)
        padded_vertices.append(vertex)
        padded_edges.append(edge)
        padded_shape_types.append(shape_type)

    return {
        'grid': torch.stack(padded_grids),
        'instance_map': torch.stack(padded_instances),
        'vertex_map': torch.stack(padded_vertices),
        'edge_map': torch.stack(padded_edges),
        'shape_type_map': torch.stack(padded_shape_types),
        'original_sizes': original_sizes,
    }


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    original_sizes: List[tuple],
    focal_loss_fn: FocalLoss,
    loss_weights: Dict[str, float]
):
    """
    Compute multi-task loss.

    Args:
        outputs: model predictions
        targets: ground truth
        original_sizes: list of (h, w) tuples for each sample
        focal_loss_fn: focal loss for vertices/edges
        loss_weights: dict of loss weights for each task
    """
    batch_size = len(original_sizes)
    device = outputs['instances'].device

    total_loss = 0.0
    losses = {}

    for i in range(batch_size):
        h, w = original_sizes[i]

        # Crop to original size
        inst_pred = outputs['instances'][i:i+1, :, :h, :w]
        inst_target = targets['instance_map'][i:i+1, :h, :w]

        vert_pred = outputs['vertices'][i:i+1, :, :h, :w]
        vert_target = targets['vertex_map'][i:i+1, :, :h, :w]

        edge_pred = outputs['edges'][i:i+1, :, :h, :w]
        edge_target = targets['edge_map'][i:i+1, :, :h, :w]

        shape_pred = outputs['shape_classes'][i:i+1, :, :h, :w]
        shape_target = targets['shape_type_map'][i:i+1, :h, :w]

        # Instance segmentation loss (cross-entropy)
        instance_loss = F.cross_entropy(inst_pred, inst_target)

        # Vertex detection loss (focal loss for class imbalance)
        vertex_loss = focal_loss_fn(vert_pred, vert_target)

        # Edge detection loss (focal loss for class imbalance)
        edge_loss = focal_loss_fn(edge_pred, edge_target)

        # Shape classification loss (cross-entropy on non-background pixels)
        # Only compute loss on pixels that belong to a shape (non-background)
        non_bg = shape_target > 0
        if non_bg.sum() > 0:
            shape_loss = F.cross_entropy(
                shape_pred[:, :, non_bg[0]],
                shape_target[non_bg],
                reduction='mean'
            )
        else:
            shape_loss = torch.tensor(0.0, device=device)

        # Weighted sum
        sample_loss = (
            loss_weights['instance'] * instance_loss +
            loss_weights['vertex'] * vertex_loss +
            loss_weights['edge'] * edge_loss +
            loss_weights['shape_class'] * shape_loss
        )

        total_loss += sample_loss

        # Track individual losses
        if i == 0:
            losses['instance'] = instance_loss.item()
            losses['vertex'] = vertex_loss.item()
            losses['edge'] = edge_loss.item()
            losses['shape_class'] = shape_loss.item()

    return total_loss / batch_size, losses


def compute_metrics(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    original_sizes: List[tuple]
):
    """Compute evaluation metrics."""
    batch_size = len(original_sizes)
    metrics = {
        'instance_acc': 0.0,
        'vertex_acc': 0.0,
        'edge_acc': 0.0,
        'shape_acc': 0.0,
    }

    for i in range(batch_size):
        h, w = original_sizes[i]

        # Instance accuracy
        inst_pred = outputs['instances'][i, :, :h, :w].argmax(dim=0)
        inst_target = targets['instance_map'][i, :h, :w]
        metrics['instance_acc'] += (inst_pred == inst_target).float().mean().item()

        # Vertex accuracy (threshold at 0.5)
        vert_pred = (outputs['vertices'][i, 0, :h, :w] > 0.5).float()
        vert_target = targets['vertex_map'][i, 0, :h, :w]
        metrics['vertex_acc'] += (vert_pred == vert_target).float().mean().item()

        # Edge accuracy (threshold at 0.5)
        edge_pred = (outputs['edges'][i, 0, :h, :w] > 0.5).float()
        edge_target = targets['edge_map'][i, 0, :h, :w]
        metrics['edge_acc'] += (edge_pred == edge_target).float().mean().item()

        # Shape accuracy (only on non-background)
        shape_pred = outputs['shape_classes'][i, :, :h, :w].argmax(dim=0)
        shape_target = targets['shape_type_map'][i, :h, :w]
        non_bg = shape_target > 0
        if non_bg.sum() > 0:
            metrics['shape_acc'] += (shape_pred[non_bg] == shape_target[non_bg]).float().mean().item()

    # Average over batch
    for key in metrics:
        metrics[key] /= batch_size

    return metrics


def train_epoch(model, dataloader, optimizer, focal_loss_fn, loss_weights, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_metrics = {
        'instance_acc': 0.0,
        'vertex_acc': 0.0,
        'edge_acc': 0.0,
        'shape_acc': 0.0,
    }

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Move to device
        grid = batch['grid'].to(device)
        targets = {
            'instance_map': batch['instance_map'].to(device),
            'vertex_map': batch['vertex_map'].to(device),
            'edge_map': batch['edge_map'].to(device),
            'shape_type_map': batch['shape_type_map'].to(device),
        }
        original_sizes = batch['original_sizes']

        # Forward
        optimizer.zero_grad()
        outputs = model(grid)

        # Compute loss
        loss, losses = compute_loss(
            outputs, targets, original_sizes, focal_loss_fn, loss_weights
        )

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            metrics = compute_metrics(outputs, targets, original_sizes)

        total_loss += loss.item()
        for key in total_metrics:
            total_metrics[key] += metrics[key]

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'inst_acc': f"{metrics['instance_acc']:.3f}",
            'vert_acc': f"{metrics['vertex_acc']:.3f}",
            'shape_acc': f"{metrics['shape_acc']:.3f}",
        })

    # Average over epoch
    total_loss /= len(dataloader)
    for key in total_metrics:
        total_metrics[key] /= len(dataloader)

    return total_loss, total_metrics


def validate(model, dataloader, focal_loss_fn, loss_weights, device):
    """Validate on validation set."""
    model.eval()
    total_loss = 0.0
    total_metrics = {
        'instance_acc': 0.0,
        'vertex_acc': 0.0,
        'edge_acc': 0.0,
        'shape_acc': 0.0,
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            # Move to device
            grid = batch['grid'].to(device)
            targets = {
                'instance_map': batch['instance_map'].to(device),
                'vertex_map': batch['vertex_map'].to(device),
                'edge_map': batch['edge_map'].to(device),
                'shape_type_map': batch['shape_type_map'].to(device),
            }
            original_sizes = batch['original_sizes']

            # Forward
            outputs = model(grid)

            # Compute loss
            loss, losses = compute_loss(
                outputs, targets, original_sizes, focal_loss_fn, loss_weights
            )

            # Metrics
            metrics = compute_metrics(outputs, targets, original_sizes)

            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += metrics[key]

    # Average
    total_loss /= len(dataloader)
    for key in total_metrics:
        total_metrics[key] /= len(dataloader)

    return total_loss, total_metrics


def main():
    # Configuration
    NUM_COLORS = 10
    MAX_INSTANCES = 10
    TRAIN_SAMPLES = 10000
    VAL_SAMPLES = 1000
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3

    # Balanced loss weights (no extreme multipliers!)
    LOSS_WEIGHTS = {
        'instance': 1.0,
        'vertex': 1.5,      # Slightly higher (was 2.0 with 10x class weight)
        'edge': 1.0,
        'shape_class': 1.5,  # Moderately higher (was 3.0)
    }

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # Generate or load datasets
    print("\nGenerating training data...")
    train_dataset = generate_dataset(
        num_samples=TRAIN_SAMPLES,
        num_colors=NUM_COLORS,
        min_size=5,
        max_size=30,
        cache_path='data/train_dataset_unet.pkl'
    )

    print("Generating validation data...")
    val_dataset = generate_dataset(
        num_samples=VAL_SAMPLES,
        num_colors=NUM_COLORS,
        min_size=5,
        max_size=30,
        cache_path='data/val_dataset_unet.pkl'
    )

    # Create datasets
    train_data = ShapeDataset(train_dataset, augment=True, max_instances=MAX_INSTANCES)
    val_data = ShapeDataset(val_dataset, augment=False, max_instances=MAX_INSTANCES)

    # Create dataloaders
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_variable_size
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_variable_size
    )

    # Create model
    print("\nCreating model...")
    model = CompactShapeUNet(
        num_colors=NUM_COLORS,
        max_instances=MAX_INSTANCES,
        num_shape_classes=NUM_SHAPE_CLASSES,
        bilinear=True
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Focal loss for vertex/edge detection
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*60}")

        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, focal_loss_fn, LOSS_WEIGHTS, DEVICE
        )

        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, focal_loss_fn, LOSS_WEIGHTS, DEVICE
        )

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Metrics: inst={train_metrics['instance_acc']:.3f}, "
              f"vert={train_metrics['vertex_acc']:.3f}, "
              f"edge={train_metrics['edge_acc']:.3f}, "
              f"shape={train_metrics['shape_acc']:.3f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Metrics: inst={val_metrics['instance_acc']:.3f}, "
              f"vert={val_metrics['vertex_acc']:.3f}, "
              f"edge={val_metrics['edge_acc']:.3f}, "
              f"shape={val_metrics['shape_acc']:.3f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, 'checkpoints/unet_best.pth')
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoints/unet_epoch_{epoch+1}.pth')

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    main()

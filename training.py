"""
Training Pipeline for Shape Recognition
Handles multi-task loss computation, metrics, and training loop
"""

import torch
import torch.nn.functional as F


class ShapeRecognitionTrainer:
    """Training pipeline for shape recognition."""

    def __init__(self, model, device=None):
        if device is None:
            # Prefer MPS (Apple Silicon) > CUDA > CPU
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

    def compute_loss(self, outputs, targets, original_sizes):
        """Compute multi-task loss."""
        batch_size = len(original_sizes)

        instance_loss = 0
        vertex_loss = 0
        edge_loss = 0

        for i in range(batch_size):
            h, w = original_sizes[i]

            # Crop to original size
            inst_pred = outputs['instances'][i, :, :h, :w]
            inst_target = targets['instance_map'][i, :h, :w]

            vert_pred = outputs['vertices'][i, 0, :h, :w]
            vert_target = targets['vertex_map'][i, :h, :w]

            edge_pred = outputs['edges'][i, 0, :h, :w]
            edge_target = targets['edge_map'][i, :h, :w]

            # Instance segmentation loss (cross entropy)
            instance_loss += F.cross_entropy(inst_pred.unsqueeze(0),
                                            inst_target.unsqueeze(0))

            # Vertex detection loss (BCE with class weighting)
            pos_weight = torch.tensor([10.0]).to(self.device)  # Vertices are rare
            vertex_loss += F.binary_cross_entropy(vert_pred, vert_target,
                                                 weight=1 + vert_target * 9)

            # Edge detection loss (BCE with class weighting)
            edge_loss += F.binary_cross_entropy(edge_pred, edge_target,
                                               weight=1 + edge_target * 4)

        # Average over batch
        instance_loss /= batch_size
        vertex_loss /= batch_size
        edge_loss /= batch_size

        # Weighted combination
        total_loss = instance_loss + 2.0 * vertex_loss + 1.5 * edge_loss

        return {
            'total': total_loss,
            'instance': instance_loss,
            'vertex': vertex_loss,
            'edge': edge_loss
        }

    def compute_metrics(self, outputs, targets, original_sizes):
        """Compute evaluation metrics."""
        batch_size = len(original_sizes)
        metrics = {
            'vertex_precision': 0,
            'vertex_recall': 0,
            'edge_iou': 0,
            'instance_accuracy': 0
        }

        for i in range(batch_size):
            h, w = original_sizes[i]

            # Vertex metrics
            vert_pred = (outputs['vertices'][i, 0, :h, :w] > 0.5).float()
            vert_target = targets['vertex_map'][i, :h, :w]

            tp = (vert_pred * vert_target).sum()
            fp = (vert_pred * (1 - vert_target)).sum()
            fn = ((1 - vert_pred) * vert_target).sum()

            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)

            metrics['vertex_precision'] += precision.item()
            metrics['vertex_recall'] += recall.item()

            # Edge IoU
            edge_pred = (outputs['edges'][i, 0, :h, :w] > 0.5).float()
            edge_target = targets['edge_map'][i, :h, :w]

            intersection = (edge_pred * edge_target).sum()
            union = ((edge_pred + edge_target) > 0).float().sum()
            iou = intersection / (union + 1e-6)

            metrics['edge_iou'] += iou.item()

            # Instance accuracy
            inst_pred = torch.argmax(outputs['instances'][i, :, :h, :w], dim=0)
            inst_target = targets['instance_map'][i, :h, :w]
            accuracy = (inst_pred == inst_target).float().mean()

            metrics['instance_accuracy'] += accuracy.item()

        # Average over batch
        for key in metrics:
            metrics[key] /= batch_size

        return metrics

    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        loss_components = {'instance': 0, 'vertex': 0, 'edge': 0}

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            grid = batch['grid'].to(self.device)
            instance_map = batch['instance_map'].to(self.device)
            vertex_map = batch['vertex_map'].to(self.device)
            edge_map = batch['edge_map'].to(self.device)

            targets = {
                'instance_map': instance_map,
                'vertex_map': vertex_map,
                'edge_map': edge_map
            }

            # Forward pass
            outputs = self.model(grid)
            losses = self.compute_loss(outputs, targets, batch['original_sizes'])

            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Accumulate losses
            total_loss += losses['total'].item()
            for key in loss_components:
                loss_components[key] += losses[key].item()

        # Average losses
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches

        return avg_loss, loss_components

    def validate(self, dataloader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_metrics = {
            'vertex_precision': 0,
            'vertex_recall': 0,
            'edge_iou': 0,
            'instance_accuracy': 0
        }

        with torch.no_grad():
            for batch in dataloader:
                grid = batch['grid'].to(self.device)
                instance_map = batch['instance_map'].to(self.device)
                vertex_map = batch['vertex_map'].to(self.device)
                edge_map = batch['edge_map'].to(self.device)

                targets = {
                    'instance_map': instance_map,
                    'vertex_map': vertex_map,
                    'edge_map': edge_map
                }

                outputs = self.model(grid)
                losses = self.compute_loss(outputs, targets, batch['original_sizes'])
                metrics = self.compute_metrics(outputs, targets, batch['original_sizes'])

                total_loss += losses['total'].item()
                for key in all_metrics:
                    all_metrics[key] += metrics[key]

        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        for key in all_metrics:
            all_metrics[key] /= num_batches

        return avg_loss, all_metrics

    def train(self, train_loader, val_loader, num_epochs=50):
        """Full training loop."""
        best_val_loss = float('inf')
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }

        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print("-" * 80)

        for epoch in range(num_epochs):
            # Train
            train_loss, loss_components = self.train_epoch(train_loader)

            # Validate
            val_loss, val_metrics = self.validate(val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)

            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} "
                  f"[Inst: {loss_components['instance']:.4f}, "
                  f"Vert: {loss_components['vertex']:.4f}, "
                  f"Edge: {loss_components['edge']:.4f}]")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Metrics: "
                  f"V-Prec: {val_metrics['vertex_precision']:.3f}, "
                  f"V-Rec: {val_metrics['vertex_recall']:.3f}, "
                  f"Edge-IoU: {val_metrics['edge_iou']:.3f}, "
                  f"Inst-Acc: {val_metrics['instance_accuracy']:.3f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                print("  ✓ Best model saved")

            print("-" * 80)

        return history

"""
Training Pipeline with Keypoint Regression
"""

import torch
import torch.nn.functional as F


class KeypointTrainer:
    """Training pipeline with keypoint regression loss."""

    def __init__(self, model, device=None, save_path='best_model_keypoints.pth'):
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.model = model.to(device)
        self.device = device
        self.save_path = save_path
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=7
        )

    def compute_loss(self, outputs, targets, original_sizes):
        """
        Compute multi-task loss including keypoint regression.

        Args:
            outputs: Dict with 'instances', 'shape_classes', 'keypoints'
            targets: Dict with 'instance_map', 'shape_type_map', 'keypoint_targets'
            original_sizes: List of (H, W) tuples for each sample in batch

        Returns:
            Dict of losses
        """
        batch_size = len(original_sizes)

        instance_loss = 0
        shape_class_loss = 0
        keypoint_loss = 0

        for i in range(batch_size):
            h, w = original_sizes[i]

            # Crop to original size
            inst_pred = outputs['instances'][i, :, :h, :w]
            inst_target = targets['instance_map'][i, :h, :w]

            shape_pred = outputs['shape_classes'][i, :, :h, :w]
            shape_target = targets['shape_type_map'][i, :h, :w]

            # Instance segmentation loss
            instance_loss += F.cross_entropy(inst_pred.unsqueeze(0),
                                            inst_target.unsqueeze(0))

            # Shape classification loss
            shape_class_loss += F.cross_entropy(shape_pred.unsqueeze(0),
                                               shape_target.unsqueeze(0))

            # Keypoint regression loss
            kp_pred = outputs['keypoints'][i]  # [max_instances, max_keypoints, 3]
            kp_target = targets['keypoint_targets'][i]  # [max_instances, max_keypoints, 3]

            kp_loss_sample = self._compute_keypoint_loss(kp_pred, kp_target)
            keypoint_loss += kp_loss_sample

        # Average over batch
        instance_loss /= batch_size
        shape_class_loss /= batch_size
        keypoint_loss /= batch_size

        # Weighted combination
        # Keypoint loss gets highest weight since it's our main focus
        total_loss = (instance_loss +
                     2.0 * shape_class_loss +
                     5.0 * keypoint_loss)  # HIGH weight for keypoints

        return {
            'total': total_loss,
            'instance': instance_loss,
            'shape_class': shape_class_loss,
            'keypoint': keypoint_loss
        }

    def _compute_keypoint_loss(self, kp_pred, kp_target):
        """
        Compute keypoint loss for a single sample.

        Args:
            kp_pred: [max_instances, max_keypoints, 3] predicted (y, x, valid)
            kp_target: [max_instances, max_keypoints, 3] target (y, x, valid)

        Returns:
            Scalar loss
        """
        # Validity mask: only compute loss for valid keypoints
        valid_mask = kp_target[:, :, 2]  # [max_instances, max_keypoints]

        # Coordinate loss (Smooth L1 for robustness)
        coord_pred = kp_pred[:, :, :2]  # [max_instances, max_keypoints, 2]
        coord_target = kp_target[:, :, :2]

        # Only compute loss for valid keypoints
        coord_loss = F.smooth_l1_loss(
            coord_pred[valid_mask > 0.5],
            coord_target[valid_mask > 0.5],
            reduction='mean'
        ) if (valid_mask > 0.5).sum() > 0 else torch.tensor(0.0, device=kp_pred.device)

        # Validity loss (binary cross-entropy)
        valid_pred = kp_pred[:, :, 2]  # [max_instances, max_keypoints]
        valid_target = kp_target[:, :, 2]

        valid_loss = F.binary_cross_entropy(valid_pred, valid_target, reduction='mean')

        # Combine losses
        total_kp_loss = coord_loss + 0.5 * valid_loss

        return total_kp_loss

    def compute_metrics(self, outputs, targets, original_sizes):
        """Compute evaluation metrics including keypoint accuracy."""
        batch_size = len(original_sizes)
        metrics = {
            'instance_accuracy': 0,
            'shape_class_accuracy': 0,
            'keypoint_accuracy': 0,  # Mean distance error
            'keypoint_validity_acc': 0
        }

        for i in range(batch_size):
            h, w = original_sizes[i]

            # Instance accuracy
            inst_pred = torch.argmax(outputs['instances'][i, :, :h, :w], dim=0)
            inst_target = targets['instance_map'][i, :h, :w]
            accuracy = (inst_pred == inst_target).float().mean()
            metrics['instance_accuracy'] += accuracy.item()

            # Shape classification accuracy
            shape_pred = torch.argmax(outputs['shape_classes'][i, :, :h, :w], dim=0)
            shape_target = targets['shape_type_map'][i, :h, :w]

            # Only evaluate on non-background pixels
            non_bg = shape_target > 0
            if non_bg.sum() > 0:
                shape_acc = (shape_pred[non_bg] == shape_target[non_bg]).float().mean()
                metrics['shape_class_accuracy'] += shape_acc.item()

            # Keypoint accuracy: mean distance error (in pixels)
            kp_pred = outputs['keypoints'][i]  # [max_instances, max_keypoints, 3]
            kp_target = targets['keypoint_targets'][i]

            valid_mask = kp_target[:, :, 2] > 0.5  # Valid keypoints

            if valid_mask.sum() > 0:
                # Denormalize coordinates
                coords_pred = kp_pred[:, :, :2].clone()
                coords_target = kp_target[:, :, :2].clone()

                coords_pred[:, :, 0] *= h  # y
                coords_pred[:, :, 1] *= w  # x
                coords_target[:, :, 0] *= h
                coords_target[:, :, 1] *= w

                # Compute distance error for valid keypoints
                dist = torch.norm(coords_pred[valid_mask] - coords_target[valid_mask], dim=1)
                mean_dist = dist.mean()
                metrics['keypoint_accuracy'] += mean_dist.item()

                # Validity accuracy
                valid_pred = kp_pred[:, :, 2] > 0.5
                valid_target = kp_target[:, :, 2] > 0.5
                valid_acc = (valid_pred == valid_target).float().mean()
                metrics['keypoint_validity_acc'] += valid_acc.item()

        # Average over batch
        for key in metrics:
            metrics[key] /= batch_size

        return metrics

    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        loss_components = {'instance': 0, 'shape_class': 0, 'keypoint': 0}

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            grid = batch['grid'].to(self.device)
            instance_map = batch['instance_map'].to(self.device)
            shape_type_map = batch['shape_type_map'].to(self.device)
            keypoint_targets = batch['keypoint_targets'].to(self.device)

            targets = {
                'instance_map': instance_map,
                'shape_type_map': shape_type_map,
                'keypoint_targets': keypoint_targets
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
            'instance_accuracy': 0,
            'shape_class_accuracy': 0,
            'keypoint_accuracy': 0,
            'keypoint_validity_acc': 0
        }

        with torch.no_grad():
            for batch in dataloader:
                grid = batch['grid'].to(self.device)
                instance_map = batch['instance_map'].to(self.device)
                shape_type_map = batch['shape_type_map'].to(self.device)
                keypoint_targets = batch['keypoint_targets'].to(self.device)

                targets = {
                    'instance_map': instance_map,
                    'shape_type_map': shape_type_map,
                    'keypoint_targets': keypoint_targets
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

    def train(self, train_loader, val_loader, num_epochs=50, early_stopping_patience=10):
        """Full training loop with early stopping."""
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }

        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
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
                  f"ShapeClass: {loss_components['shape_class']:.4f}, "
                  f"Keypoint: {loss_components['keypoint']:.4f}]")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Metrics: "
                  f"Inst-Acc: {val_metrics['instance_accuracy']:.3f}, "
                  f"Shape-Acc: {val_metrics['shape_class_accuracy']:.3f}, "
                  f"KP-Err: {val_metrics['keypoint_accuracy']:.2f}px, "
                  f"KP-Valid: {val_metrics['keypoint_validity_acc']:.3f}")

            # Save best model and check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f"  ✓ Best model saved to {self.save_path}")
            else:
                epochs_without_improvement += 1
                print(f"  No improvement for {epochs_without_improvement} epoch(s)")

            # Early stopping check
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n⊗ Early stopping triggered after {epoch+1} epochs")
                print(f"  No improvement for {early_stopping_patience} consecutive epochs")
                print(f"  Best validation loss: {best_val_loss:.4f}")
                break

            print("-" * 80)

        return history

"""
Deep Learning Model for DAiSEE Dataset - Video Emotion Recognition
Uses 3D CNN architecture for spatiotemporal feature extraction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# logging.basicConfig(level=logging.INFO)  <-- Removed to prevent conflict
logger = logging.getLogger(__name__)


class VideoEmotionDataset(Dataset):
    """PyTorch Dataset for preprocessed video samples"""
    
    def __init__(self, samples: List[Dict]):
        """
        Args:
            samples: List of preprocessed samples from VideoPreprocessor
        """
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = sample['frames']  # Shape: (T, H, W, 3)
        labels = sample['labels']
        
        # Convert to (3, T, H, W) for PyTorch 3D CNN
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
        
        # Create multi-label target
        target = torch.tensor([
            labels['Boredom'],
            labels['Engagement'],
            labels['Confusion'],
            labels['Frustration']
        ], dtype=torch.long)
        
        return frames, target


class Conv3DBlock(nn.Module):
    """3D Convolution block with batch norm and activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class C3D(nn.Module):
    """
    3D CNN for video classification.
    Suitable for spatiotemporal feature learning from videos.
    """
    
    def __init__(self, num_classes: int = 4, pretrained: bool = False):
        """
        Args:
            num_classes: Number of emotion classes (4 for DAiSEE)
            pretrained: Whether to use pretrained weights (from Sports-1M)
        """
        super().__init__()
        self.num_classes = num_classes
        
        # Input: (B, 3, T, H, W)
        
        # Block 1
        self.conv1 = Conv3DBlock(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # Block 2
        self.conv2 = Conv3DBlock(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Block 3
        self.conv3a = Conv3DBlock(128, 256, kernel_size=3, padding=1)
        self.conv3b = Conv3DBlock(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Block 4
        self.conv4a = Conv3DBlock(256, 512, kernel_size=3, padding=1)
        self.conv4b = Conv3DBlock(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Block 5
        self.conv5a = Conv3DBlock(512, 512, kernel_size=3, padding=1)
        self.conv5b = Conv3DBlock(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # FC layers
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 3, T, H, W)
        
        Returns:
            Class logits of shape (B, num_classes)
        """
        # Block 1
        x = self.conv1(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)
        
        # Block 4
        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.pool4(x)
        
        # Block 5
        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.pool5(x)
        
        # Classification head
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class VideoTrainer:
    """Trainer for video emotion recognition model"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        """
        Args:
            model: PyTorch model
            device: Device to use (cpu or cuda)
            learning_rate: Learning rate
            weight_decay: L2 regularization
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Multi-label loss (one loss per emotion class)
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_frames, batch_labels in train_loader:
            batch_frames = batch_frames.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_frames)  # (B, 4)
            
            # Multi-label loss: compute loss for each emotion class
            loss = 0
            for class_idx in range(4):
                loss += self.criterion(outputs[:, class_idx:class_idx+1], batch_labels[:, class_idx])
            loss = loss / 4
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * batch_frames.size(0)
            
            # Accuracy per class (for monitoring)
            preds = torch.argmax(outputs, dim=1)
            total_samples += batch_frames.size(0)
        
        avg_loss = total_loss / total_samples
        
        return {'loss': avg_loss}
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        # Per-class accuracy
        class_correct = [0] * 4
        class_total = [0] * 4
        
        with torch.no_grad():
            for batch_frames, batch_labels in val_loader:
                batch_frames = batch_frames.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_frames)
                
                # Multi-label loss
                loss = 0
                for class_idx in range(4):
                    loss += self.criterion(outputs[:, class_idx:class_idx+1], batch_labels[:, class_idx])
                loss = loss / 4
                
                total_loss += loss.item() * batch_frames.size(0)
                total_samples += batch_frames.size(0)
                
                # Per-class accuracy
                for class_idx in range(4):
                    preds = torch.argmax(outputs[:, class_idx:class_idx+1], dim=1)
                    class_correct[class_idx] += (preds == batch_labels[:, class_idx]).sum().item()
                    class_total[class_idx] += batch_frames.size(0)
        
        avg_loss = total_loss / total_samples
        class_names = ['Boredom', 'Engagement', 'Confusion', 'Frustration']
        metrics = {'loss': avg_loss}
        
        for i, name in enumerate(class_names):
            if class_total[i] > 0:
                metrics[f'{name}_acc'] = class_correct[i] / class_total[i]
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        save_best: bool = True,
        checkpoint_dir: str = 'checkpoints',
    ):
        """
        Train model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            save_best: Whether to save best model
            checkpoint_dir: Directory to save checkpoints
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        best_loss = float('inf')
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}"
            )
            
            # Log per-class accuracies
            for key in val_metrics:
                if key != 'loss':
                    logger.info(f"  {key}: {val_metrics[key]:.4f}")
            
            # Save best model
            if save_best and val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                torch.save(
                    self.model.state_dict(),
                    Path(checkpoint_dir) / 'best_model.pth'
                )
                logger.info(f"  -> Saved best model")


if __name__ == '__main__':
    # Example: Initialize and display model
    model = C3D(num_classes=4)
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

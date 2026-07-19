"""
Google Colab Training Notebook for DAiSEE Video Emotion Recognition
This script loads preprocessed .npy chunks and trains a deep learning model.

Instructions:
1. Upload preprocessed_videos/ folder to Google Drive
2. Copy this script into a Colab notebook
3. Mount Google Drive and run cells sequentially
"""

# ============================================================================
# CELL 1: Install dependencies (Colab already has most packages)
# ============================================================================

# Install PyTorch with GPU support
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# ============================================================================
# CELL 2: Mount Google Drive and load preprocessed data
# ============================================================================

from google.colab import drive
import os
import numpy as np
import pandas as pd

drive.mount('/content/drive')

# Path to preprocessed data in Google Drive
preprocessed_path = '/content/drive/MyDrive/preprocessed_videos'

# Load metadata CSV
metadata_df = pd.read_csv(os.path.join(preprocessed_path, 'metadata.csv'))
print(f"Loaded metadata with {len(metadata_df)} videos")
print("\nMetadata preview:")
print(metadata_df.head())

# Check available chunks
chunks = sorted([f for f in os.listdir(preprocessed_path) if f.startswith('X_chunk_')])
print(f"\nFound {len(chunks)} preprocessed chunks")
for chunk in chunks[:5]:  # Show first 5
    print(f"  {chunk}")


# ============================================================================
# CELL 3: Create PyTorch Dataset and DataLoader
# ============================================================================

import torch
from torch.utils.data import Dataset, DataLoader

class VideoChunkDataset(Dataset):
    """Load video chunks from disk on-the-fly"""
    
    def __init__(self, metadata_df, chunks_dir, split='train'):
        """
        Args:
            metadata_df: DataFrame with video metadata
            chunks_dir: Directory containing .npy files
            split: 'train', 'val', or 'test' (stratified split)
        """
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.chunks_dir = chunks_dir
        self.split = split
        
        # Simple split: 70% train, 15% val, 15% test
        total = len(self.metadata_df)
        train_split = int(0.7 * total)
        val_split = int(0.85 * total)
        
        if split == 'train':
            self.indices = range(0, train_split)
        elif split == 'val':
            self.indices = range(train_split, val_split)
        else:  # test
            self.indices = range(val_split, total)
        
        self.metadata_df = self.metadata_df.iloc[self.indices].reset_index(drop=True)
        self._load_chunks_cache()
    
    def _load_chunks_cache(self):
        """Cache loaded chunks to speed up training"""
        self.chunks_cache = {}
        unique_chunks = self.metadata_df['chunk_number'].unique()
        
        print(f"Loading {len(unique_chunks)} chunks for {self.split} set...")
        for chunk_num in unique_chunks:
            X_path = os.path.join(self.chunks_dir, f'X_chunk_{chunk_num}.npy')
            Y_path = os.path.join(self.chunks_dir, f'Y_chunk_{chunk_num}.npy')
            
            X = np.load(X_path).astype(np.float32)  # (B, 20, 224, 224, 3)
            Y = np.load(Y_path).astype(np.float32)  # (B, 4)
            
            self.chunks_cache[chunk_num] = (X, Y)
        print(f"Loaded all chunks for {self.split} set")
    
    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        chunk_num = int(row['chunk_number'])
        idx_in_chunk = int(row['index_in_chunk'])
        
        # Load from cache
        X, Y = self.chunks_cache[chunk_num]
        
        frames = X[idx_in_chunk]  # (20, 224, 224, 3)
        labels = Y[idx_in_chunk]  # (4,)
        
        # Convert to PyTorch format: (C, T, H, W)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2)  # (3, 20, 224, 224)
        labels = torch.from_numpy(labels).long()
        
        return frames, labels


# Create datasets
train_dataset = VideoChunkDataset(metadata_df, preprocessed_path, split='train')
val_dataset = VideoChunkDataset(metadata_df, preprocessed_path, split='val')
test_dataset = VideoChunkDataset(metadata_df, preprocessed_path, split='test')

print(f"\nDataset sizes:")
print(f"  Train: {len(train_dataset)}")
print(f"  Val: {len(val_dataset)}")
print(f"  Test: {len(test_dataset)}")

# Create DataLoaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"\nDataLoader batches:")
print(f"  Train: {len(train_loader)}")
print(f"  Val: {len(val_loader)}")
print(f"  Test: {len(test_loader)}")


# ============================================================================
# CELL 4: Define 3D CNN Model (C3D)
# ============================================================================

import torch.nn as nn

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class C3D(nn.Module):
    """3D CNN for video emotion recognition"""
    
    def __init__(self, num_classes=4):
        super().__init__()
        
        # Input: (B, 3, 20, 224, 224)
        
        self.conv1 = Conv3DBlock(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = Conv3DBlock(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3a = Conv3DBlock(128, 256, kernel_size=3, padding=1)
        self.conv3b = Conv3DBlock(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv4a = Conv3DBlock(256, 512, kernel_size=3, padding=1)
        self.conv4b = Conv3DBlock(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv5a = Conv3DBlock(512, 512, kernel_size=3, padding=1)
        self.conv5b = Conv3DBlock(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)
        
        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.pool4(x)
        
        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.pool5(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = C3D(num_classes=4).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel initialized on {device}")
print(f"Total parameters: {total_params:,}")


# ============================================================================
# CELL 5: Training Loop
# ============================================================================

import torch.optim as optim
from tqdm import tqdm

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_frames, batch_labels in tqdm(train_loader, desc="Training"):
        batch_frames = batch_frames.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(batch_frames)  # (B, 4)
        
        # Multi-label loss: compute loss for each emotion class
        criterion = nn.CrossEntropyLoss()
        loss = 0
        for class_idx in range(4):
            loss += criterion(outputs, batch_labels[:, class_idx])
        loss = loss / 4
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    class_correct = [0] * 4
    class_total = [0] * 4
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_frames, batch_labels in tqdm(val_loader, desc="Evaluating"):
            batch_frames = batch_frames.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_frames)
            
            loss = 0
            for class_idx in range(4):
                loss += criterion(outputs, batch_labels[:, class_idx])
            loss = loss / 4
            
            total_loss += loss.item()
            
            for class_idx in range(4):
                preds = torch.argmax(outputs, dim=1)
                class_correct[class_idx] += (preds == batch_labels[:, class_idx]).sum().item()
                class_total[class_idx] += batch_frames.size(0)
    
    avg_loss = total_loss / len(val_loader)
    class_accs = [class_correct[i] / class_total[i] for i in range(4)]
    
    return avg_loss, class_accs


# Training configuration
num_epochs = 30
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train
print("\nStarting training...")
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss, class_accs = evaluate(model, val_loader, device)
    
    class_names = ['Boredom', 'Engagement', 'Confusion', 'Frustration']
    acc_str = ' | '.join([f"{name}: {acc:.3f}" for name, acc in zip(class_names, class_accs)])
    
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"  {acc_str}")


# ============================================================================
# CELL 6: Test Set Evaluation
# ============================================================================

test_loss, test_accs = evaluate(model, test_loader, device)
class_names = ['Boredom', 'Engagement', 'Confusion', 'Frustration']

print("\n" + "="*60)
print("TEST SET RESULTS")
print("="*60)
print(f"Test Loss: {test_loss:.4f}\n")

for name, acc in zip(class_names, test_accs):
    print(f"{name:20s}: {acc:.4f}")

# Save model
model_save_path = '/content/drive/MyDrive/c3d_emotion_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"\nModel saved to: {model_save_path}")

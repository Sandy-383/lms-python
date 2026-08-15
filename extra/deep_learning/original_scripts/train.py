"""
Complete training pipeline for DAiSEE video emotion recognition.
Demonstrates end-to-end usage of video preprocessing and model training.
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from video_preprocessing import VideoPreprocessor, DataGenerator
from model import VideoEmotionDataset, C3D, VideoTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Configuration
    dataset_root = 'archive/DAiSEE/DataSet'
    labels_root = 'archive/DAiSEE/Labels'
    output_dir = 'preprocessed_data'
    checkpoint_dir = 'checkpoints'
    
    # Hyperparameters
    frame_size = (224, 224)
    frames_per_video = 30
    batch_size = 8
    learning_rate = 1e-3
    epochs = 50
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Step 1: Preprocess videos
    logger.info("="*80)
    logger.info("STEP 1: VIDEO PREPROCESSING")
    logger.info("="*80)
    
    preprocessor = VideoPreprocessor(
        dataset_root=dataset_root,
        labels_file=f'{labels_root}/AllLabels.csv',
        frame_size=frame_size,
        frames_per_video=frames_per_video,
    )
    
    # Preprocess training set
    logger.info("\nPreprocessing Training Set...")
    train_samples = preprocessor.preprocess_dataset(
        split='Train',
        output_dir=None,  # Set to output_dir to save preprocessed videos
        normalize=True,
        augment=False,  # Set to True for data augmentation
        compute_flow=False,  # Set to True for optical flow features
        save_npy=False,  # Set to True to save to disk
    )
    
    # Preprocess validation set
    logger.info("\nPreprocessing Validation Set...")
    val_samples = preprocessor.preprocess_dataset(
        split='Validation',
        normalize=True,
        augment=False,
        compute_flow=False,
    )
    
    # Preprocess test set
    logger.info("\nPreprocessing Test Set...")
    test_samples = preprocessor.preprocess_dataset(
        split='Test',
        normalize=True,
        augment=False,
        compute_flow=False,
    )
    
    logger.info(f"\nDataset Summary:")
    logger.info(f"  Train samples: {len(train_samples)}")
    logger.info(f"  Validation samples: {len(val_samples)}")
    logger.info(f"  Test samples: {len(test_samples)}")
    
    # Step 2: Create data loaders
    logger.info("\n" + "="*80)
    logger.info("STEP 2: CREATE DATA LOADERS")
    logger.info("="*80)
    
    # Create PyTorch datasets
    train_dataset = VideoEmotionDataset(train_samples)
    val_dataset = VideoEmotionDataset(val_samples)
    test_dataset = VideoEmotionDataset(test_samples)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Increase based on your CPU cores
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Data loaders created")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    
    # Step 3: Initialize model
    logger.info("\n" + "="*80)
    logger.info("STEP 3: INITIALIZE MODEL")
    logger.info("="*80)
    
    model = C3D(num_classes=4)
    logger.info(f"Model: C3D")
    logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  Device: {device}")
    
    # Step 4: Train model
    logger.info("\n" + "="*80)
    logger.info("STEP 4: TRAIN MODEL")
    logger.info("="*80)
    
    trainer = VideoTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
    )
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_best=True,
        checkpoint_dir=checkpoint_dir,
    )
    
    # Step 5: Evaluate on test set
    logger.info("\n" + "="*80)
    logger.info("STEP 5: EVALUATE ON TEST SET")
    logger.info("="*80)
    
    # Load best model
    best_model_path = Path(checkpoint_dir) / 'best_model.pth'
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"Loaded best model from {best_model_path}")
    
    test_metrics = trainer.evaluate(test_loader)
    logger.info("\nTest Set Results:")
    for key, value in test_metrics.items():
        logger.info(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    main()

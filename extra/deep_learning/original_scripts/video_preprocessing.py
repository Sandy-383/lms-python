"""
Video Preprocessing Module for DAiSEE Dataset
Handles frame extraction, optical flow, and data augmentation for video classification.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoPreprocessor:
    """
    Preprocesses video clips from the DAiSEE dataset.
    Extracts frames and optical flow for temporal modeling.
    """
    
    def __init__(
        self,
        dataset_root: str,
        labels_file: str,
        frame_size: Tuple[int, int] = (224, 224),
        frames_per_video: int = 30,
        stride: int = 1,
    ):
        """
        Initialize the video preprocessor.
        
        Args:
            dataset_root: Root path to DAiSEE dataset
            labels_file: Path to labels CSV file
            frame_size: Target frame resolution (height, width)
            frames_per_video: Number of frames to extract per video
            stride: Frame sampling stride (1 = every frame, 2 = every 2nd frame, etc.)
        """
        self.dataset_root = Path(dataset_root)
        self.labels_file = Path(labels_file)
        self.frame_size = frame_size
        self.frames_per_video = frames_per_video
        self.stride = stride
        
        # Load labels
        self.labels_df = pd.read_csv(self.labels_file)
        self.label_columns = ['Boredom', 'Engagement', 'Confusion', 'Frustration']
        
        logger.info(f"Loaded {len(self.labels_df)} labels from {self.labels_file}")
    
    def get_video_path(self, clip_id: str, split: str = 'Train') -> Optional[Path]:
        """
        Locate a video file given its clip ID.
        
        Args:
            clip_id: Video file name (e.g., '1100011002.avi')
            split: Dataset split ('Train', 'Test', 'Validation')
        
        Returns:
            Path to video file or None if not found
        """
        subject_id = clip_id[:6]  # First 6 digits = subject ID
        video_path = self.dataset_root / split / subject_id / clip_id
        
        if video_path.exists():
            return video_path
        
        # Try without extension if full path doesn't exist
        for ext in ['.avi', '.mp4', '.mov']:
            alt_path = self.dataset_root / split / subject_id / (clip_id.replace('.avi', ext))
            if alt_path.exists():
                return alt_path
        
        return None
    
    def extract_frames(
        self,
        video_path: str,
        num_frames: Optional[int] = None,
        resize: bool = True,
    ) -> np.ndarray:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract (uses self.frames_per_video if None)
            resize: Whether to resize frames to self.frame_size
        
        Returns:
            Array of shape (num_frames, height, width, 3) with dtype uint8
        """
        num_frames = num_frames or self.frames_per_video
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0:
                logger.warning(f"Video {video_path} has 0 frames")
                cap.release()
                return None
            
            # Calculate frame indices to extract
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            frames = []
            frame_count = 0
            target_frame_idx = 0
            
            while cap.isOpened() and target_frame_idx < len(frame_indices):
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame_count == frame_indices[target_frame_idx]:
                    if resize:
                        frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
                    frames.append(frame)
                    target_frame_idx += 1
                
                frame_count += 1
            
            cap.release()
            
            if len(frames) == 0:
                logger.warning(f"No frames extracted from {video_path}")
                return None
            
            # Pad with last frame if needed
            while len(frames) < num_frames:
                frames.append(frames[-1])
            
            return np.array(frames[:num_frames], dtype=np.uint8)
        
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            return None
    
    def compute_optical_flow(
        self,
        frames: np.ndarray,
        method: str = 'farneback'
    ) -> np.ndarray:
        """
        Compute optical flow between consecutive frames.
        
        Args:
            frames: Array of shape (num_frames, height, width, 3)
            method: Optical flow method ('farneback', 'lucaskanade')
        
        Returns:
            Array of shape (num_frames-1, height, width, 2) representing flow
        """
        if frames is None or len(frames) < 2:
            return None
        
        gray_frames = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
        optical_flows = []
        
        if method == 'farneback':
            for i in range(len(gray_frames) - 1):
                flow = cv2.calcOpticalFlowFarneback(
                    gray_frames[i],
                    gray_frames[i + 1],
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    n8neighbors=True,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0
                )
                optical_flows.append(flow)
        
        return np.array(optical_flows, dtype=np.float32) if optical_flows else None
    
    def normalize_frames(self, frames: np.ndarray, method: str = 'imagenet') -> np.ndarray:
        """
        Normalize frames using ImageNet or standard normalization.
        
        Args:
            frames: Array of shape (num_frames, height, width, 3) with values [0, 255]
            method: 'imagenet' or 'standard' (mean=127.5, std=127.5)
        
        Returns:
            Normalized frames
        """
        if frames is None:
            return None
        
        frames = frames.astype(np.float32)
        
        if method == 'imagenet':
            # ImageNet normalization (for RGB)
            mean = np.array([123.675, 116.28, 103.53])
            std = np.array([58.395, 57.12, 57.375])
            frames = (frames - mean) / std
        else:
            # Standard normalization
            frames = (frames - 127.5) / 127.5
        
        return frames
    
    def augment_frames(
        self,
        frames: np.ndarray,
        rotation_range: float = 10,
        brightness_range: float = 0.2,
        zoom_range: float = 0.1,
        horizontal_flip: bool = True,
    ) -> np.ndarray:
        """
        Apply data augmentation to frames.
        
        Args:
            frames: Array of shape (num_frames, height, width, 3)
            rotation_range: Rotation angle in degrees
            brightness_range: Brightness adjustment range
            zoom_range: Zoom range
            horizontal_flip: Whether to apply random horizontal flip
        
        Returns:
            Augmented frames
        """
        if frames is None:
            return None
        
        augmented = frames.copy().astype(np.float32)
        
        # Random horizontal flip
        if horizontal_flip and np.random.rand() > 0.5:
            augmented = np.array([cv2.flip(f, 1) for f in augmented])
        
        # Random brightness
        if np.random.rand() > 0.5:
            brightness_factor = np.random.uniform(1 - brightness_range, 1 + brightness_range)
            augmented = np.clip(augmented * brightness_factor, 0, 255)
        
        # Random rotation
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-rotation_range, rotation_range)
            h, w = augmented.shape[1:3]
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            augmented = np.array([
                cv2.warpAffine(f.astype(np.uint8), matrix, (w, h))
                for f in augmented
            ]).astype(np.float32)
        
        return augmented
    
    def preprocess_video(
        self,
        video_path: str,
        labels: Dict[str, int],
        normalize: bool = True,
        augment: bool = False,
        compute_flow: bool = False,
    ) -> Dict:
        """
        Complete preprocessing pipeline for a single video.
        
        Args:
            video_path: Path to video file
            labels: Dictionary with emotion labels
            normalize: Whether to normalize frames
            augment: Whether to apply data augmentation
            compute_flow: Whether to compute optical flow
        
        Returns:
            Dictionary containing processed video data and labels
        """
        # Extract frames
        frames = self.extract_frames(video_path)
        if frames is None:
            return None
        
        # Apply augmentation
        if augment:
            frames = self.augment_frames(frames)
        
        # Compute optical flow
        optical_flow = None
        if compute_flow:
            optical_flow = self.compute_optical_flow(frames)
        
        # Normalize
        if normalize:
            frames = self.normalize_frames(frames)
        
        return {
            'frames': frames,
            'optical_flow': optical_flow,
            'labels': labels,
            'video_path': str(video_path),
        }
    
    def preprocess_dataset(
        self,
        split: str = 'Train',
        output_dir: Optional[str] = None,
        normalize: bool = True,
        augment: bool = False,
        compute_flow: bool = False,
        save_npy: bool = False,
    ) -> List[Dict]:
        """
        Preprocess entire dataset split.
        
        Args:
            split: Dataset split ('Train', 'Test', 'Validation')
            output_dir: Directory to save preprocessed data
            normalize: Whether to normalize frames
            augment: Whether to apply data augmentation
            compute_flow: Whether to compute optical flow
            save_npy: Whether to save to .npy files
        
        Returns:
            List of preprocessed video samples
        """
        output_path = Path(output_dir) if output_dir else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
        
        processed_samples = []
        failed_videos = []
        
        # Load split-specific labels if available
        split_labels_file = self.dataset_root / 'Labels' / f'{split}Labels.csv'
        if split_labels_file.exists():
            split_df = pd.read_csv(split_labels_file)
        else:
            split_df = self.labels_df
        
        logger.info(f"Processing {split} split with {len(split_df)} videos")
        
        for idx, row in tqdm(split_df.iterrows(), total=len(split_df)):
            clip_id = row['ClipID']
            video_path = self.get_video_path(clip_id, split)
            
            if video_path is None:
                failed_videos.append(clip_id)
                continue
            
            # Extract labels
            labels = {col: int(row[col]) for col in self.label_columns}
            
            # Preprocess video
            sample = self.preprocess_video(
                str(video_path),
                labels,
                normalize=normalize,
                augment=augment,
                compute_flow=compute_flow,
            )
            
            if sample is not None:
                processed_samples.append(sample)
                
                # Save to disk if requested
                if save_npy and output_path:
                    save_name = clip_id.replace('.avi', '')
                    np.save(output_path / f'{save_name}_frames.npy', sample['frames'])
                    if sample['optical_flow'] is not None:
                        np.save(output_path / f'{save_name}_flow.npy', sample['optical_flow'])
        
        logger.info(f"Successfully processed {len(processed_samples)} videos")
        if failed_videos:
            logger.warning(f"Failed to process {len(failed_videos)} videos: {failed_videos[:10]}")
        
        return processed_samples


class DataGenerator:
    """
    PyTorch-compatible data generator for video samples.
    """
    
    def __init__(self, samples: List[Dict], batch_size: int = 8, shuffle: bool = True):
        """
        Initialize data generator.
        
        Args:
            samples: List of preprocessed samples
            batch_size: Batch size
            shuffle: Whether to shuffle data
        """
        self.samples = samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(samples))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))
    
    def __iter__(self):
        """Iterate through batches"""
        for batch_idx in range(len(self)):
            batch_indices = self.indices[
                batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size
            ]
            
            batch_frames = []
            batch_labels = []
            
            for idx in batch_indices:
                sample = self.samples[self.indices[idx]]
                batch_frames.append(sample['frames'])
                batch_labels.append([
                    sample['labels']['Boredom'],
                    sample['labels']['Engagement'],
                    sample['labels']['Confusion'],
                    sample['labels']['Frustration'],
                ])
            
            yield np.array(batch_frames), np.array(batch_labels)


if __name__ == '__main__':
    # Example usage
    dataset_root = 'archive/DAiSEE/DataSet'
    labels_file = 'archive/DAiSEE/Labels/AllLabels.csv'
    
    # Initialize preprocessor
    preprocessor = VideoPreprocessor(
        dataset_root=dataset_root,
        labels_file=labels_file,
        frame_size=(224, 224),
        frames_per_video=30,
    )
    
    # Preprocess training set
    train_samples = preprocessor.preprocess_dataset(
        split='Train',
        normalize=True,
        augment=False,
        compute_flow=False,
    )
    
    print(f"Total training samples: {len(train_samples)}")
    if train_samples:
        sample = train_samples[0]
        print(f"Sample frames shape: {sample['frames'].shape}")
        print(f"Sample labels: {sample['labels']}")

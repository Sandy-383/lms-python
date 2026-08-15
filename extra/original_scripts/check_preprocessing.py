"""
Script to continue preprocessing from where it left off
and generate metadata CSV from existing chunks.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

def generate_metadata_from_chunks(preprocessed_dir='preprocessed_videos'):
    """
    Generate metadata CSV from existing .npy chunks.
    Useful when preprocessing was interrupted.
    """
    metadata_records = []
    video_index = 0
    
    # Find all X_chunk files
    chunk_files = sorted([f for f in os.listdir(preprocessed_dir) if f.startswith('X_chunk_')])
    
    print(f"Generating metadata from {len(chunk_files)} chunks...")
    
    for chunk_file in chunk_files:
        chunk_num = int(chunk_file.split('_')[2].split('.')[0])
        
        # Load the chunk to get its shape
        X_path = os.path.join(preprocessed_dir, chunk_file)
        Y_path = os.path.join(preprocessed_dir, f'Y_chunk_{chunk_num}.npy')
        
        if not os.path.exists(Y_path):
            print(f"Warning: {Y_path} not found, skipping {chunk_file}")
            continue
        
        X_chunk = np.load(X_path)
        Y_chunk = np.load(Y_path)
        
        num_videos_in_chunk = len(X_chunk)
        
        # Create metadata for each video in this chunk
        for idx_in_chunk in range(num_videos_in_chunk):
            labels = Y_chunk[idx_in_chunk]
            
            metadata_records.append({
                'video_index': video_index,
                'clip_id': f'video_{video_index:06d}.avi',  # Generic ID since we lost original mapping
                'chunk_number': chunk_num,
                'index_in_chunk': idx_in_chunk,
                'label_boredom': int(labels[0]),
                'label_engagement': int(labels[1]),
                'label_confusion': int(labels[2]),
                'label_frustration': int(labels[3]),
                'video_path': '',  # Unknown since we lost original paths
            })
            
            video_index += 1
    
    # Save metadata CSV
    metadata_df = pd.DataFrame(metadata_records)
    metadata_csv_path = os.path.join(preprocessed_dir, 'metadata.csv')
    metadata_df.to_csv(metadata_csv_path, index=False)
    
    print(f"\nMetadata CSV created!")
    print(f"  Total videos: {len(metadata_records)}")
    print(f"  Path: {metadata_csv_path}")
    print(f"\nMetadata preview:")
    print(metadata_df.head(10))
    
    return metadata_df


def check_preprocessing_status(preprocessed_dir='preprocessed_videos', labels_file='archive/DAiSEE/Labels/AllLabels.csv'):
    """
    Check how many videos have been preprocessed vs total.
    """
    # Load all labels
    df_labels = pd.read_csv(labels_file)
    df_labels.columns = df_labels.columns.str.strip()
    total_videos = len(df_labels)
    
    # Count preprocessed videos
    chunk_files = [f for f in os.listdir(preprocessed_dir) if f.startswith('X_chunk_')]
    
    if not chunk_files:
        preprocessed_count = 0
    else:
        preprocessed_count = 0
        for chunk_file in chunk_files:
            chunk_num = int(chunk_file.split('_')[2].split('.')[0])
            X_path = os.path.join(preprocessed_dir, chunk_file)
            X_chunk = np.load(X_path)
            preprocessed_count += len(X_chunk)
    
    print("="*60)
    print("PREPROCESSING STATUS")
    print("="*60)
    print(f"Total videos to process: {total_videos}")
    print(f"Videos preprocessed:     {preprocessed_count}")
    print(f"Remaining:               {total_videos - preprocessed_count}")
    print(f"Progress:                {100 * preprocessed_count / total_videos:.1f}%")
    print("="*60)
    
    return preprocessed_count, total_videos


if __name__ == '__main__':
    print("Preprocessing Status & Metadata Generation\n")
    
    # Check status
    preprocessed_count, total_videos = check_preprocessing_status()
    
    # Generate metadata from existing chunks
    print("\nGenerating metadata from existing chunks...")
    metadata_df = generate_metadata_from_chunks()
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. To continue preprocessing, run:")
    print("   python preprocess.py")
    print("\n2. The preprocessed_videos/ folder now has:")
    print("   - metadata.csv (tracks all preprocessed videos)")
    print("   - X_chunk_*.npy (video frames)")
    print("   - Y_chunk_*.npy (emotion labels)")
    print("\n3. Ready to upload to Google Colab!")
    print("="*60)

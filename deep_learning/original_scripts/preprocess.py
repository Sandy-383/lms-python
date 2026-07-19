import cv2
import numpy as np
import pandas as pd
import os
import gc # Garbage Collector
import re # Regular Expressions
from pathlib import Path

# --- CONFIGURATION (Ensure these global variables are defined before calling the function) ---
IMG_SIZE = 112  # Reduced from 224 for faster processing
SEQ_LENGTH = 10  # Reduced from 20 for faster processing
FACE_PAD = 20

# Setup Face Detection (Load once at startup)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("ERROR: Failed to load face cascade. Install opencv-python-headless or ensure haarcascades are available.")
    exit(1)

# --- HELPER FUNCTIONS (You should define these above the main function) ---
def crop_and_preprocess_frame(frame, face_cascade, img_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4) 
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        x1, y1 = max(0, x - FACE_PAD), max(0, y - FACE_PAD)
        x2, y2 = min(frame.shape[1], x + w + FACE_PAD), min(frame.shape[0], y + h + FACE_PAD)
        cropped_face = frame[y1:y2, x1:x2]
    else:
        # Center crop fallback if no face is detected
        h, w = frame.shape[:2]
        crop_h = crop_w = min(h, w, img_size) 
        cropped_face = frame[h//2 - crop_h//2 : h//2 + crop_h//2, w//2 - crop_w//2 : w//2 + crop_w//2]

    try:
        # Resize and normalize
        return cv2.resize(cropped_face, (img_size, img_size)) / 255.0
    except Exception:
        # Return black frame on error
        return np.zeros((img_size, img_size, 3))

def preprocess_video(video_path, face_cascade, img_size, seq_length):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(int(total_frames / seq_length), 1)

    for i in range(seq_length):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip)
        ret, frame = cap.read()
        if ret:
            frames.append(crop_and_preprocess_frame(frame, face_cascade, img_size))
        else:
            frames.append(np.zeros((img_size, img_size, 3)))
    cap.release()
    return np.array(frames)


# --- MAIN FUNCTION ---

def batch_process_and_save(labels_file, video_root, output_path, batch_size):
    """
    Processes video data in chunks, performs face cropping, and saves 
    the resulting features to disk to manage RAM limits.
    Also creates a metadata CSV for tracking all preprocessed videos.
    """
    df = pd.read_csv(labels_file)
    
    # FIX for KeyError: 'Frustration' — Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # 1. Map all video files for fast lookup
    print("Mapping all video files (Recursive search)...")
    video_paths = {}
    for root, dirs, files in os.walk(video_root):
        for file in files:
            if file.endswith(".avi") or file.endswith(".mp4"):
                # Use a cleaner version of filename for map key (base name only)
                base_name = os.path.splitext(file)[0]
                video_paths[base_name] = os.path.join(root, file)

    print(f"Found {len(video_paths)} potential video files.")
    
    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 2. Iterate through CSV in batches
    X_batch, Y_batch = [], []
    batch_count = 0
    total_processed = 0
    metadata_records = []  # Track all processed videos

    for index, row in df.iterrows():
        clip_id_with_ext = row['ClipID'] 
        clip_id_base = os.path.splitext(clip_id_with_ext)[0]
        
        # Check if the video file exists in our map
        if clip_id_base in video_paths:
            video_path = video_paths[clip_id_base]
            
            # Extract the 4 labels (now guaranteed to be clean)
            labels = [row['Boredom'], row['Engagement'], row['Confusion'], row['Frustration']]
            
            try:
                frames = preprocess_video(video_path, face_cascade, IMG_SIZE, SEQ_LENGTH)
                X_batch.append(frames)
                Y_batch.append(labels)
                
                # Record metadata for this video
                metadata_records.append({
                    'video_index': total_processed,
                    'clip_id': clip_id_with_ext,
                    'chunk_number': batch_count + 1,  # Will be updated after batch save
                    'index_in_chunk': len(X_batch) - 1,
                    'label_boredom': labels[0],
                    'label_engagement': labels[1],
                    'label_confusion': labels[2],
                    'label_frustration': labels[3],
                    'video_path': video_path,
                })
                total_processed += 1
                
                # --- SAVE BATCH AND CLEAR MEMORY ---
                if len(X_batch) >= batch_size:
                    batch_count += 1
                    print(f"--- Saving Batch {batch_count} ({total_processed} total processed) ---")
                    
                    np.save(os.path.join(output_path, f'X_chunk_{batch_count}.npy'), np.array(X_batch))
                    np.save(os.path.join(output_path, f'Y_chunk_{batch_count}.npy'), np.array(Y_batch))
                    
                    # Update chunk numbers for this batch's metadata
                    for i in range(len(X_batch)):
                        metadata_records[total_processed - len(X_batch) + i]['chunk_number'] = batch_count
                    
                    # Clear memory for the next batch
                    X_batch, Y_batch = [], []
                    gc.collect() # Explicitly clear RAM

            except Exception as e:
                print(f"Error processing {clip_id_with_ext}: {e}")
        
    # 3. Save the final partial batch
    if len(X_batch) > 0:
        batch_count += 1
        print(f"--- Saving Final Batch {batch_count} ({total_processed} total processed) ---")
        np.save(os.path.join(output_path, f'X_chunk_{batch_count}.npy'), np.array(X_batch))
        np.save(os.path.join(output_path, f'Y_chunk_{batch_count}.npy'), np.array(Y_batch))
        
        # Update chunk numbers for this batch's metadata
        for i in range(len(X_batch)):
            metadata_records[total_processed - len(X_batch) + i]['chunk_number'] = batch_count

    # 4. Save metadata CSV
    metadata_df = pd.DataFrame(metadata_records)
    metadata_csv_path = os.path.join(output_path, 'metadata.csv')
    metadata_df.to_csv(metadata_csv_path, index=False)
    print(f"\nMetadata saved to: {metadata_csv_path}")
    
    print(f"\nPreprocessing complete. Total videos processed: {total_processed}")
    print(f"Features saved as {batch_count} chunks to {output_path}")
    print(f"Metadata CSV created with {len(metadata_records)} entries")


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Configuration - Modify as needed
    labels_file = 'archive/DAiSEE/Labels/AllLabels.csv'
    video_root = 'archive/DAiSEE/DataSet'
    output_path = 'preprocessed_videos'
    batch_size = 10  # Number of videos per batch before saving
    
    # Validate paths exist
    if not os.path.exists(labels_file):
        print(f"ERROR: Labels file not found: {labels_file}")
        print(f"Current working directory: {os.getcwd()}")
        exit(1)
    
    if not os.path.exists(video_root):
        print(f"ERROR: Video root directory not found: {video_root}")
        print(f"Current working directory: {os.getcwd()}")
        exit(1)
    
    print(f"Starting preprocessing...")
    print(f"  Labels: {labels_file}")
    print(f"  Videos: {video_root}")
    print(f"  Output: {output_path}")
    print(f"  Batch size: {batch_size}")
    print(f"  Frame size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Frames per video: {SEQ_LENGTH}")
    print()
    
    # Run preprocessing
    batch_process_and_save(labels_file, video_root, output_path, batch_size)
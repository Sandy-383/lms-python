import os
import numpy as np
from pathlib import Path

pre_dir = Path('preprocessed_videos')
chunk_files = sorted(pre_dir.glob('X_chunk_*.npy'))

print(f'Found {len(chunk_files)} chunks to convert')

for i, f in enumerate(chunk_files, 1):
    out_f = f.with_suffix('.u8.npy')
    if out_f.exists():
        print(f'[{i}/{len(chunk_files)}] Skipping {f.name} (already converted)')
        continue
    try:
        arr = np.load(f, mmap_mode='r')
        # Detect range and dtype
        dtype = arr.dtype
        arr_max = float(arr.max())
        arr_min = float(arr.min())
        print(f'[{i}/{len(chunk_files)}] Converting {f.name} dtype={dtype} min={arr_min:.4f} max={arr_max:.4f}')
        
        # Load into memory chunk-by-chunk if large
        loaded = np.array(arr, dtype=np.float32)
        # If values are in [0,1], scale to 0-255
        if arr_max <= 1.01:
            converted = np.rint(loaded * 255.0).astype(np.uint8)
        else:
            # assume already in 0-255 range
            converted = np.rint(loaded).clip(0,255).astype(np.uint8)
        
        # Save as uint8 .npy (smaller) - no compression for speed
        np.save(out_f, converted)
        print(f'    Saved -> {out_f.name} size={out_f.stat().st_size/1024/1024:.2f} MB')
    except Exception as e:
        print(f'[{i}/{len(chunk_files)}] ERROR converting {f.name}: {e}')

print('Conversion complete. New files have suffix .u8.npy')
print('Review a few converted files, then if OK you can delete original .npy and rename .u8.npy back to .npy')
print('To delete originals after verification:')
print('  for f in preprocessed_videos\\*.npy: if f.endswith(".u8.npy")==False then remove')

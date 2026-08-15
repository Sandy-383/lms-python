import os
import numpy as np
from pathlib import Path

pre_dir = Path('preprocessed_videos')
files = sorted(pre_dir.glob('X_chunk_*.npy'))

print(f'Found {len(files)} X_chunk files in {pre_dir}')

total_bytes = 0

for i, f in enumerate(files[:20]):  # show first 20 for brevity
    try:
        arr = np.load(f, mmap_mode='r')
        shape = arr.shape
        dtype = arr.dtype
        bytesize = os.path.getsize(f)
        total_bytes += bytesize
        print(f'{i+1:02d}. {f.name} | shape={shape} | dtype={dtype} | size={bytesize/1024/1024:.2f} MB')
    except Exception as e:
        print(f'{i+1:02d}. {f.name} | ERROR loading: {e}')

# summary across all files
all_files = list(pre_dir.glob('X_chunk_*.npy'))
sum_size = sum(os.path.getsize(x) for x in all_files)
print('\nSummary:')
print(f'  Total X_chunk files: {len(all_files)}')
print(f'  Total size (X_chunk): {sum_size/1024/1024:.2f} MB')

# show Y_chunk small files
y_files = list(pre_dir.glob('Y_chunk_*.npy'))
sum_y = sum(os.path.getsize(x) for x in y_files)
print(f'  Total Y_chunk files: {len(y_files)}')
print(f'  Total size (Y_chunk): {sum_y/1024/1024:.2f} MB')
print(f'  Combined size: {(sum_size+sum_y)/1024/1024:.2f} MB')

# check dtypes distribution
from collections import Counter

dtypes = Counter()
shapes = Counter()
for f in all_files:
    try:
        arr = np.load(f, mmap_mode='r')
        dtypes[str(arr.dtype)] += 1
        shapes[str(arr.shape)] += 1
    except Exception:
        dtypes['error'] += 1

print('\nDtype distribution (X_chunks):')
for k,v in dtypes.items():
    print(f'  {k}: {v}')

print('\nShape samples (top 5):')
for k,v in shapes.most_common(5):
    print(f'  {k}: {v}')

print('\nInspector finished.')

import numpy as np

print('Testing file access...')
try:
    with np.load('data/converted/E003_all_markers_merged.npz') as f:
        print("Sample labels shape:", f['label'].shape)
        # Check if you have multiple markers per sample
        print("First few labels:", f['label'][:5])
        print("Label dimensions:", f['label'].ndim)
except Exception as e:
    print('✗ Error:', e)
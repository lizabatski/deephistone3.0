import numpy as np

print("Loading full .npz file (this may take a bit)...")
with np.load("data/converted/E003_all_markers_merged.npz", mmap_mode=None) as f:
    keys = f['keys'][:100]
    dna = f['dna'][:100]
    dnase = f['dnase'][:100]
    label = f['label'][:100]

print("Saving mini version...")
np.savez("data/converted/mini_merged.npz", keys=keys, dna=dna, dnase=dnase, label=label)
print("Done.")

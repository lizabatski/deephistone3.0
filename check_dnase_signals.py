import numpy as np
import sys

if len(sys.argv) != 2:
    print("Usage: python check_dnase_zeros.py <converted_dataset.npz>")
    sys.exit(1)

data_file = sys.argv[1]
print(f"🔍 Checking DNase signal in: {data_file}")

with np.load(data_file, mmap_mode='r') as f:
    dnase = f["dnase"]  # shape: (N, 1, 1, 1000)
    keys = f["keys"]

    total = dnase.shape[0]
    flat_dnase = dnase.reshape(total, 1000)

    zero_mask = np.all(flat_dnase == 0, axis=1)
    num_all_zero = np.sum(zero_mask)
    num_nonzero = total - num_all_zero

    print(f"\n📊 DNase Signal Summary:")
    print(f"  Total samples:        {total:,}")
    print(f"  All-zero DNase:       {num_all_zero:,} ({num_all_zero / total:.1%})")
    print(f"  Non-zero DNase:       {num_nonzero:,} ({num_nonzero / total:.1%})")

    if num_all_zero > 0:
        print("\n🧬 Example keys with all-zero DNase:")
        for i in np.where(zero_mask)[0][:5]:
            print(f"  {keys[i]}")

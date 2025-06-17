import numpy as np

def verify_dnase_alignment(npz_file, num_samples_to_check=5):
    print("=" * 60)
    print(f"VERIFYING DNase-DNA ALIGNMENT IN: {npz_file}")
    print("=" * 60)

    try:
        with np.load(npz_file, mmap_mode="r") as f:
            print("File opened with memory-mapped read")
            keys = f["keys"]
            dna_file = f["dna"]
            dnase_file = f["dnase"]

            for i in range(min(num_samples_to_check, len(keys))):
                print(f"\nSample {i+1} - Key: {keys[i].decode() if isinstance(keys[i], bytes) else keys[i]}")
                dna = dna_file[i]   # shape: (1000, 4)
                dnase = dnase_file[i]  # shape: (1000,)

                # Simple stats
                avg_dnase = np.mean(dnase)
                max_dnase = np.max(dnase)
                print(f"  Avg DNase signal: {avg_dnase:.3f}, Max: {max_dnase:.3f}")

                base_frequencies = [np.mean(dna[:, j]) for j in range(4)]
                base_labels = ["A", "C", "G", "T"]
                freqs_str = ", ".join([f"{base_labels[j]}: {base_frequencies[j]*100:.1f}%" for j in range(4)])
                print(f"  DNA base frequencies: {freqs_str}")

                # Flags for interpretation
                if avg_dnase < 0.1:
                    print("  ⚠️  DNase signal is low — region likely inaccessible.")
                elif max_dnase > 1.0:
                    print("  ✅  Strong DNase peak detected — likely open chromatin.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify DNase-DNA alignment in NPZ file")
    parser.add_argument("npz_file", help="Path to the .npz file")
    parser.add_argument("--num", type=int, default=5, help="Number of samples to check (default: 5)")
    args = parser.parse_args()

    verify_dnase_alignment(args.npz_file, args.num)

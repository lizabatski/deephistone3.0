import numpy as np
import argparse
import matplotlib.pyplot as plt

def inspect_dnase(npz_file, num_samples=5, show_plot=True):
    print("=" * 60)
    print(f"INSPECTING DNase SIGNAL IN: {npz_file}")
    print("=" * 60)

    with np.load(npz_file, allow_pickle=True) as f:
        keys = f["keys"]
        dnase = f["dnase"]

        print(f"Total samples: {len(dnase)}")

        for i in range(min(num_samples, len(dnase))):
            signal = dnase[i]
            key = keys[i].decode() if isinstance(keys[i], bytes) else keys[i]
            print(f"\nSample {i}: {key}")
            print(f"  DNase shape: {signal.shape}")
            print(f"  Min: {np.min(signal):.4f}, Max: {np.max(signal):.4f}, Mean: {np.mean(signal):.4f}")

            if show_plot:
                plt.figure(figsize=(10, 2))
                plt.plot(signal.squeeze(), color='black')
                plt.title(f"DNase signal - Sample {i} ({key})")
                plt.xlabel("Position (bp)")
                plt.ylabel("DNase Openness")
                plt.tight_layout()
                plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect DNase signal from a DeepHistone .npz file")
    parser.add_argument("npz_file", help="Path to .npz file")
    parser.add_argument("--num", type=int, default=5, help="Number of samples to inspect")
    parser.add_argument("--no-plot", action="store_true", help="Suppress matplotlib plots")
    args = parser.parse_args()

    inspect_dnase(args.npz_file, args.num, not args.no_plot)

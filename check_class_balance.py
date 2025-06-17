import numpy as np
import sys

def quick_class_balance_check(data_file):
    """Minimal check focused on class balance - the main cause of low auPRC"""
    print("=" * 60)
    print("QUICK CLASS BALANCE CHECK")
    print("=" * 60)

    try:
        with np.load(data_file) as f:
            print("File opened successfully")
            print(f"Keys in file: {list(f.keys())}")

            print("\nLoading labels only...")
            labels = f['label'][:]
            print(f"Original label shape: {labels.shape}")

            # Remove extra dimensions like (N, 1, 7)
            labels = np.squeeze(labels)
            if labels.ndim == 1:
                labels = labels.reshape(-1, 1)

            print(f"Processed label shape: {labels.shape}")

            print("\nCLASS BALANCE ANALYSIS:")
            print("This is likely why your auPRC is low if imbalanced!")
            print("-" * 40)

            num_markers = labels.shape[1]
            severe_imbalance = False

            for i in range(num_markers):
                pos_count = np.sum(labels[:, i] == 1)
                total_count = labels.shape[0]
                pos_ratio = pos_count / total_count

                print(f"Marker {i}: {pos_count:6d}/{total_count} = {pos_ratio:.1%} positive")
                if pos_ratio < 0.05 or pos_ratio > 0.95:
                    severe_imbalance = True

            print("-" * 40)
            if severe_imbalance:
                print("⚠️ Severe class imbalance detected in one or more markers!")
            else:
                print("✅ Class balance looks reasonably okay.")

    except Exception as e:
        print(f"❌ Error while processing file: {e}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python check_class_balance.py <path_to_npz_file>")
        sys.exit(1)

    data_file_path = sys.argv[1]
    print(f"Running class balance check on: {data_file_path}")
    quick_class_balance_check(data_file_path)


if __name__ == "__main__":
    main()

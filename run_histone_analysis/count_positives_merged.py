import numpy as np
from pathlib import Path

def count_positive_sites(npz_path):
    """
    Counts the number of positive sites for each of the 7 histone markers
    in a DeepHistone merged .npz file.

    Args:
        npz_path (str or Path): Path to the merged .npz file

    Returns:
        dict: Marker -> count of positive bins
    """
    npz_path = Path(npz_path)
    histone_markers = [
        'H3K4me1', 'H3K4me3', 'H3K27me3',
        'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac'
    ]

    if not npz_path.exists():
        raise FileNotFoundError(f"File not found: {npz_path}")

    print(f"Loading merged file: {npz_path}")
    data = np.load(npz_path)

    if 'label' not in data:
        raise KeyError(f"'label' key not found in {npz_path.name}")

    labels = data['label']  # shape: (N, 1, 7)
    print(f"Label shape: {labels.shape}")

    results = {}
    total = 0

    print("\nPositive Site Counts per Marker")
    print("=" * 35)
    for i, marker in enumerate(histone_markers):
        count = int(np.sum(labels[:, 0, i]))
        results[marker] = count
        total += count
        print(f"{marker:>10}: {count:,}")

    print("-" * 35)
    print(f"{'TOTAL':>10}: {total:,}")
    return results


if __name__ == "__main__":
    home_dir = Path.home()
    merged_file_path = home_dir / "deephistone" / "data" / "final" / "E005_all_markers_merged.npz"
    count_positive_sites(merged_file_path)

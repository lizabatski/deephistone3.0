from pybedtools import BedTool
import sys

# Usage: python check_peak_overlap.py <regions.bed> <peaks.bed>
if len(sys.argv) != 3:
    print("Usage: python check_peak_overlap.py <regions.bed> <peaks.bed>")
    sys.exit(1)

region_path = sys.argv[1]
peak_path = sys.argv[2]

print(f"\n🔍 Checking overlaps between:\n  Regions: {region_path}\n  Peaks: {peak_path}\n")

# Load both as BedTool objects
regions = BedTool(region_path)
peaks = BedTool(peak_path)

# Show 5 sample lines
print("📌 Sample region entries:")
print("".join([str(r) for r in regions[:5]]))

print("\n📌 Sample peak entries:")
print("".join([str(p) for p in peaks[:5]]))

# Try intersect
overlap = regions.intersect(peaks, u=True)
overlap_count = len(overlap)

print(f"\n✅ Overlaps found: {overlap_count}\n")

# Print some overlaps
if overlap_count > 0:
    print("✅ Sample overlapping regions:")
    print("".join([str(r) for r in overlap[:5]]))
else:
    print("⚠️ No overlaps found. Double-check genome builds, coordinate systems, and chromosome filters.")

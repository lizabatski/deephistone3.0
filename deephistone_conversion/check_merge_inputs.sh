#!/bin/bash

EPIGENOME="E005"
DATA_DIR="data/converted/${EPIGENOME}"
PATTERN="${DATA_DIR}/${EPIGENOME}_*_expected_format.npz"

echo "Checking for input files in: $DATA_DIR"
echo "Pattern: $PATTERN"

# Check directory
if [ ! -d "$DATA_DIR" ]; then
    echo " ERROR: Directory '$DATA_DIR' does not exist."
    exit 1
fi

# List files
echo " Matching files:"
ls -lh $PATTERN 2>/dev/null

# Count files
file_count=$(ls $PATTERN 2>/dev/null | wc -l)
echo " Found $file_count files"

# Check expected count
if [ "$file_count" -ne 7 ]; then
    echo " ERROR: Expected 7 files, but found $file_count."
    echo "Please make sure all 7 histone marker files are present."
    exit 1
else
    echo " All expected input files are present."
fi

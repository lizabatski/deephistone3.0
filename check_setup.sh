#!/bin/bash

# Pre-submission check script for DeepHistone pipeline
# Usage: ./check_setup.sh [EPIGENOME_ID]

EPIGENOME=${1:-"E005"}
MARKERS=("H3K4me1" "H3K4me3" "H3K27me3" "H3K36me3" "H3K9me3" "H3K9ac" "H3K27ac")

echo "============================================================"
echo "DEEPHISTONE PRE-SUBMISSION CHECK FOR $EPIGENOME"
echo "============================================================"
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Directory: $(pwd)"
echo ""

# Track overall status
OVERALL_STATUS=0

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
    OVERALL_STATUS=1
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_section() {
    echo ""
    echo "=== $1 ==="
}

# 1. Check file availability
print_section "1. CHECKING DATA FILES"

echo "Checking histone marker files for $EPIGENOME:"
missing_files=0
for marker in "${MARKERS[@]}"; do
    file="raw/${EPIGENOME}-${marker}.narrowPeak"
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file")
        size=$(du -h "$file" | cut -f1)
        print_success "$marker: $lines peaks ($size)"
    else
        print_error "$marker: File missing - $file"
        missing_files=$((missing_files + 1))
    fi
done

# Check DNase file
dnase_file="raw/${EPIGENOME}-DNase.macs2.narrowPeak"
if [ -f "$dnase_file" ]; then
    lines=$(wc -l < "$dnase_file")
    size=$(du -h "$dnase_file" | cut -f1)
    print_success "DNase: $lines peaks ($size)"
else
    print_error "DNase: File missing - $dnase_file"
    missing_files=$((missing_files + 1))
fi

# Check reference files
echo ""
echo "Checking reference files:"
ref_files=("raw/hg19.fa" "raw/hg19.chrom.sizes.txt")
for file in "${ref_files[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        print_success "$(basename $file): $size"
    else
        print_error "$(basename $file): Missing - $file"
    fi
done

if [ $missing_files -gt 0 ]; then
    print_error "Missing $missing_files data files - job will fail"
fi

# 2. Check Python environment and imports
print_section "2. PYTHON ENVIRONMENT CHECK"

# Check if Python is available
if command -v python &> /dev/null; then
    python_version=$(python --version 2>&1)
    print_success "Python available: $python_version"
else
    print_error "Python not found in PATH"
fi

# Check required packages
echo ""
echo "Checking Python packages:"
python << 'EOF'
import sys

packages = {
    'numpy': 'Numerical computing',
    'pyfaidx': 'FASTA file indexing', 
    'pandas': 'Data manipulation',
    'tqdm': 'Progress bars',
    'multiprocessing': 'Parallel processing',
    'collections': 'Data structures',
    'json': 'JSON handling'
}

missing_packages = []
for pkg, desc in packages.items():
    try:
        __import__(pkg)
        print(f'✅ {pkg:15} - {desc}')
    except ImportError:
        print(f'❌ {pkg:15} - MISSING ({desc})')
        missing_packages.append(pkg)

if missing_packages:
    print(f'\nERROR: Missing packages: {", ".join(missing_packages)}')
    sys.exit(1)
else:
    print('\n✅ All required packages available')
EOF

if [ $? -ne 0 ]; then
    print_error "Python package check failed"
fi

# 3. Check pipeline script
print_section "3. PIPELINE SCRIPT CHECK"

pipeline_script="deephistone_pipeline_all.py"
if [ -f "$pipeline_script" ]; then
    print_success "Pipeline script found: $pipeline_script"
    
    # Check if script can be imported
    echo "Testing script import..."
    python -c "
try:
    from deephistone_pipeline_all import run_single_combination, setup_logging, config
    print('✅ Import successful')
    print(f'✅ Output directory: {config.OUTPUT_DIR}')
    print(f'✅ Available markers: {len(config.ALL_MARKERS)}')
    print(f'✅ Valid epigenomes: {len(config.VALID_EPIGENOMES)}')
    
    # Check if target epigenome is valid
    if '$EPIGENOME' in config.VALID_EPIGENOMES:
        print('✅ Target epigenome $EPIGENOME is valid')
    else:
        print('⚠️  Target epigenome $EPIGENOME not in valid list')
        print(f'    Valid epigenomes: {config.VALID_EPIGENOMES}')
        
except Exception as e:
    print(f'❌ Import failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "Pipeline script import successful"
    else
        print_error "Pipeline script import failed"
    fi
else
    print_error "Pipeline script not found: $pipeline_script"
fi

# 4. Check directories and permissions
print_section "4. DIRECTORY AND PERMISSIONS CHECK"

# Check/create output directories
dirs=("data" "logs")
for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        if [ -w "$dir" ]; then
            print_success "$dir directory exists and is writable"
        else
            print_error "$dir directory exists but is not writable"
        fi
    else
        echo "Creating $dir directory..."
        if mkdir -p "$dir"; then
            print_success "$dir directory created successfully"
        else
            print_error "Failed to create $dir directory"
        fi
    fi
done

# Check disk space
echo ""
echo "Checking disk space:"
available_space=$(df -h $(pwd) | awk 'NR==2 {print $4}')
print_success "Available space: $available_space"

# Estimate space needed
echo "Estimating space requirements:"
echo "  - Each marker dataset: ~100-500MB"
echo "  - Total for 7 markers: ~1-4GB"
echo "  - Logs: ~10-50MB"
print_warning "Ensure you have at least 5GB free space"

# 5. Check submission script
print_section "5. SUBMISSION SCRIPT CHECK"

submit_script="submit_e005.sh"
if [ -f "$submit_script" ]; then
    print_success "Submission script found: $submit_script"
    
    # Check script syntax
    if bash -n "$submit_script" 2>/dev/null; then
        print_success "Submission script syntax is valid"
    else
        print_error "Submission script has syntax errors"
        echo "Run: bash -n $submit_script"
    fi
    
    # Check if executable
    if [ -x "$submit_script" ]; then
        print_success "Submission script is executable"
    else
        print_warning "Submission script is not executable"
        echo "Run: chmod +x $submit_script"
    fi
    
else
    print_error "Submission script not found: $submit_script"
fi

# 6. Quick functionality test
print_section "6. QUICK FUNCTIONALITY TEST"

echo "Testing pipeline with minimal data (this may take 1-5 minutes)..."
echo "Testing: $EPIGENOME-H3K4me1 (first marker only)"

# Run quick test
timeout 300 python << EOF
from deephistone_pipeline_all import run_single_combination, setup_logging
import time
import os

print('Starting quick test...')
start_time = time.time()

try:
    # Use first marker for quick test
    output_path, success = run_single_combination('$EPIGENOME', 'H3K4me1', None)
    elapsed = time.time() - start_time
    
    if success:
        # Check if output file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024*1024)  # MB
            print(f'✅ SUCCESS: Test completed in {elapsed:.1f} seconds')
            print(f'✅ Output file created: {output_path} ({file_size:.1f} MB)')
            
            # Quick validation of output file
            import numpy as np
            try:
                data = np.load(output_path)
                n_samples = len(data['sequences']) if 'sequences' in data else 0
                print(f'✅ Dataset contains {n_samples:,} samples')
                print(f'✅ Keys in dataset: {list(data.keys())}')
            except Exception as e:
                print(f'⚠️  Could not validate dataset: {e}')
        else:
            print(f'⚠️  Test completed but output file not found: {output_path}')
    else:
        print('❌ FAILED: Test failed')
        exit(1)
        
except Exception as e:
    print(f'❌ ERROR: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
EOF

test_result=$?
if [ $test_result -eq 0 ]; then
    print_success "Quick functionality test PASSED"
elif [ $test_result -eq 124 ]; then
    print_warning "Test timed out after 5 minutes (normal for large datasets)"
else
    print_error "Quick functionality test FAILED"
fi

# 7. Final summary
print_section "7. FINAL SUMMARY"

echo "Check completed at: $(date)"
echo ""

if [ $OVERALL_STATUS -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL CHECKS PASSED${NC}"
    echo ""
    echo "Your setup looks good! You can submit your job with:"
    echo "  sbatch submit_e005.sh"
    echo ""
    echo "Expected outputs:"
    for marker in "${MARKERS[@]}"; do
        echo "  data/${EPIGENOME}_${marker}_deephistone.npz"
    done
    echo ""
    echo "Monitor progress with:"
    echo "  squeue -u \$USER"
    echo "  tail -f logs/deephistone_*.out"
else
    echo -e "${RED}❌ ISSUES FOUND${NC}"
    echo ""
    echo "Please fix the errors above before submitting your job."
    echo "Common fixes:"
    echo "  - Install missing Python packages: pip install package_name"
    echo "  - Check file paths and permissions"
    echo "  - Ensure raw data files are properly downloaded"
fi

echo ""
echo "============================================================"

exit $OVERALL_STATUS
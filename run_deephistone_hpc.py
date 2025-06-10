#!/usr/bin/env python3
# run_deephistone_hpc.py
import sys
import os
import time


sys.path.insert(0, '/home/ekourb/deephistone/scripts')


os.chdir('/home/ekourb/deephistone')

from deephistone_pipeline_all import config, setup_logging, run_single_combination

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_deephistone_hpc.py <epigenome_id>")
        sys.exit(1)
    
    epigenome_id = sys.argv[1]
    

    print(f"Working directory: {os.getcwd()}")
    print(f"Raw files location: {os.path.exists('raw')}")
    print(f"Scripts location: {os.path.exists('scripts')}")
    
    # Setup logging
    logger = setup_logging()
    
    print(f"=" * 80)
    print(f"DEEPHISTONE HPC PROCESSING: {epigenome_id}")
    print(f"=" * 80)
    print(f"Job ID: {os.environ.get('SLURM_JOB_ID', 'unknown')}")
    print(f"Node: {os.environ.get('SLURMD_NODENAME', 'unknown')}")
    print(f"CPUs: {os.environ.get('SLURM_CPUS_PER_TASK', 'unknown')}")
    print(f"Memory: {os.environ.get('SLURM_MEM_PER_NODE', 'unknown')}MB")
    print(f"=" * 80)
    
    successful = []
    failed = []
    start_time = time.time()
    
    for i, marker in enumerate(config.ALL_MARKERS, 1):
        print(f"\n--- Processing {i}/{len(config.ALL_MARKERS)}: {epigenome_id}-{marker} ---")
        marker_start = time.time()
        
        try:
            output_path, success = run_single_combination(epigenome_id, marker, logger)
            elapsed = time.time() - marker_start
            
            if success:
                file_size = os.path.getsize(output_path) / (1024*1024) if output_path else 0
                successful.append(marker)
                print(f"✓ SUCCESS: {epigenome_id}-{marker} ({elapsed/60:.1f} min, {file_size:.1f}MB)")
            else:
                failed.append(marker)
                print(f"✗ FAILED: {epigenome_id}-{marker} ({elapsed/60:.1f} min)")
                
        except Exception as e:
            elapsed = time.time() - marker_start
            failed.append(marker)
            print(f"✗ ERROR: {epigenome_id}-{marker} ({elapsed/60:.1f} min): {e}")
            logger.error(f"Error processing {epigenome_id}-{marker}: {e}")
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n" + "=" * 80)
    print(f"FINAL RESULTS: {epigenome_id}")
    print(f"=" * 80)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"Successful: {len(successful)}/{len(config.ALL_MARKERS)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\nSuccessful markers: {successful}")
    if failed:
        print(f"Failed markers: {failed}")
    
    logger.info(f"HPC job completed for {epigenome_id}: {len(successful)} successful, {len(failed)} failed")

if __name__ == "__main__":
    main()

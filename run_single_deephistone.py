# run_single_deephistone.py
import argparse
import sys
import os
import time
from pathlib import Path

# Import your existing functions
from deephistone_pipeline_all  import run_single_combination, setup_logging, config

def main():
    parser = argparse.ArgumentParser(description='Process single epigenome-marker combination')
    parser.add_argument('--epigenome', required=True, help='Epigenome ID (e.g., E003)')
    parser.add_argument('--marker', required=True, help='Histone marker (e.g., H3K4me3)')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode (chr22 only)')
    
    args = parser.parse_args()
    
    print(f"Starting processing: {args.epigenome}-{args.marker}")
    print(f"Process ID: {os.getpid()}")
    print(f"Working directory: {os.getcwd()}")
    
    # Validate inputs
    if args.epigenome not in config.VALID_EPIGENOMES:
        print(f"Error: {args.epigenome} not in valid epigenomes: {config.VALID_EPIGENOMES}")
        sys.exit(1)
    
    if args.marker not in config.ALL_MARKERS:
        print(f"Error: {args.marker} not in valid markers: {config.ALL_MARKERS}")
        sys.exit(1)
    
    # Set test mode if requested
    if args.test_mode:
        config.TEST_MODE = True
        config.TEST_CHROMOSOME = "chr22"
        print("Running in TEST MODE (chr22 only)")
    
    # Setup logging with unique filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/{args.epigenome}_{args.marker}_{timestamp}.log"
    os.makedirs("logs", exist_ok=True)
    
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Process the combination
    start_time = time.time()
    try:
        logger.info(f"Starting {args.epigenome}-{args.marker}")
        output_path, success = run_single_combination(args.epigenome, args.marker, logger)
        
        elapsed_time = time.time() - start_time
        
        if success:
            print(f"SUCCESS: Dataset saved to {output_path}")
            print(f"Processing time: {elapsed_time/3600:.2f} hours")
            logger.info(f"SUCCESS: Completed in {elapsed_time:.2f} seconds")
            sys.exit(0)
        else:
            print(f"FAILED: Processing {args.epigenome}-{args.marker}")
            logger.error(f"FAILED: Processing completed with errors")
            sys.exit(1)
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"ERROR: {e}")
        print(f"Failed after: {elapsed_time/60:.2f} minutes")
        logger.error(f"ERROR: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

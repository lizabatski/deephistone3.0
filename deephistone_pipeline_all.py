
import os
import numpy as np
from pyfaidx import Fasta
import pandas as pd
from collections import defaultdict
import time
from tqdm import tqdm
import json
from pathlib import Path
import multiprocessing as mp
from functools import partial
import logging
from datetime import datetime
import traceback


class DeepHistoneConfig:
    def __init__(self):
        # parameters from paper
        self.WINDOW_SIZE = 200  # scanning windows
        self.FINAL_WINDOW_SIZE = 1000  # final sequences for model
        self.STEP_SIZE = 200    # non-overlapping scan - KEEPING AT 200
        self.MIN_OVERLAP = 100  # minimum overlap with peak 
        self.MIN_SITES_THRESHOLD = 50000  # discard epigenomes with <50K sites
        self.RANDOM_SEED = 42
        
        #  7 histone markers from paper Table 1
        self.ALL_MARKERS = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']
        
        # The 15 epigenomes that passed their filtering (from Table 1)
        self.VALID_EPIGENOMES = [
            'E003', 'E004', 'E005', 'E006', 'E007', 'E011', 'E012', 
            'E013', 'E016', 'E024', 'E065', 'E066', 'E116', 'E117', 'E118'
        ]
        
        # paths
        self.BASE_PATH = "raw"
        self.CHROM_SIZES = "raw/hg19.chrom.sizes.txt"
        self.FASTA_PATH = "raw/hg19.fa"
        self.OUTPUT_DIR = "data"
        
        
        self.USE_MULTIPROCESSING = True
        self.N_PROCESSES = min(12, mp.cpu_count())  
        self.BATCH_SIZE = 5  # Number of datasets to process in parallel
        
        self.MAX_N_FRACTION = 0.1
        self.VALIDATE_GENOME_COVERAGE = True
        
        # Test mode settings
        self.TEST_MODE = False  
        self.TEST_CHROMOSOME = "chr22"
        
        # Batch processing options
        self.SKIP_EXISTING = True  # Skip already processed datasets
        self.CONTINUE_ON_ERROR = True  # Continue processing other datasets if one fails
        self.SAVE_PROGRESS = True  # Save progress periodically
        
        # Cache these globally to avoid repeated loading
        self._chrom_sizes = None
        self._genome = None
        
    def get_chrom_sizes(self):
        #get cached chromosome sizes
        if self._chrom_sizes is None:
            self._chrom_sizes = self._load_chromosome_sizes()
        return self._chrom_sizes
    
    def get_genome(self):
        #get cached fasta
        if self._genome is None:
            self._genome = Fasta(self.FASTA_PATH)
        return self._genome
    
    def _load_chromosome_sizes(self):
        #loading chormosomes sizes
        chrom_sizes = {}
        try:
            with open(self.CHROM_SIZES, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        chrom, size = parts[0], int(parts[1])
                        
                        if chrom in ['chrX', 'chrY']: #made the decision to not include sex chromosomes -- maybe add this back later
                            continue
                        # test mode
                        if self.TEST_MODE and chrom != self.TEST_CHROMOSOME: #currently test chromosome is chr 22 
                            continue
                        chrom_sizes[chrom] = size
        except FileNotFoundError:
            raise FileNotFoundError(f"Chromosome sizes file not found: {self.CHROM_SIZES}")
        
        if not chrom_sizes:
            raise ValueError("No chromosomes loaded - check file format and test mode settings")
        
        return chrom_sizes
    
    #reproducible path methods 
    def get_chipseq_path(self, epigenome_id, marker):
        return f"{self.BASE_PATH}/{epigenome_id}-{marker}.narrowPeak"
    
    def get_dnase_path(self, epigenome_id):
        return f"{self.BASE_PATH}/{epigenome_id}-DNase.macs2.narrowPeak"
    
    def get_output_path(self, epigenome_id, target_marker):
        suffix = f"_{self.TEST_CHROMOSOME}" if self.TEST_MODE else ""
        return f"{self.OUTPUT_DIR}/{epigenome_id}_{target_marker}_deephistone{suffix}.npz"
    
    def get_all_combinations(self):
        #all epigenome-marker combos that need to be process
        combinations = []
        for epigenome in self.VALID_EPIGENOMES:
            for marker in self.ALL_MARKERS:
                combinations.append((epigenome, marker))
        return combinations


config = DeepHistoneConfig()


def setup_logging():
    #logging for batch processing
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/deephistone_batch_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def log_progress(message, start_time=None):
    current_time = time.time()
    if start_time:
        elapsed = current_time - start_time
        print(f"[{elapsed:.2f}s] {message}")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] {message}")
    return current_time


def load_chromosome_sizes():
    #get cahced res
    return config.get_chrom_sizes()


def load_all_peaks_at_once(epigenome_id):
    #trying to optimize cuz yesterday was a fail
    start_time = log_progress(f"Loading all peaks for {epigenome_id}...")
    
    all_peaks = {marker: defaultdict(list) for marker in config.ALL_MARKERS} #there will be a dictionary where the keys are the markers and the values are dictionaries of chromosomes with lists of peaks
    
    for marker in config.ALL_MARKERS: #going through all markers
        peaks_file = config.get_chipseq_path(epigenome_id, marker)
        if os.path.exists(peaks_file):
            with open(peaks_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    cols = line.split('\t')
                    if len(cols) < 3:
                        continue
                        
                    try:
                        # col[0] is chromosome name col[1] is start position, col[2] is end position
                        chrom, start, end = cols[0], int(cols[1]), int(cols[2])
                        
                        #SEX CHROMSOME
                        if chrom in ['chrX', 'chrY']:
                           continue
                        
                        if config.TEST_MODE and chrom != config.TEST_CHROMOSOME:
                            continue
                        
                        # making sure it is within bounds
                        if start >= end or start < 0:
                            continue
                        
                        all_peaks[marker][chrom].append((start, end))
                        
                    except (ValueError, IndexError):
                        continue
    
    # sorting all peaks
    total_peaks = 0
    for marker in all_peaks:
        for chrom in all_peaks[marker]:
            all_peaks[marker][chrom].sort()
            total_peaks += len(all_peaks[marker][chrom])
    
    log_progress(f"Loaded {total_peaks:,} peaks across all markers", start_time)
    return all_peaks


def scan_chromosome_parallel(args):

    #chrom: chromosome name
    #chrom_size: size of the chromosome
    #peaks: list of tuples (start, end) for peaks in this chromosome
    #window_size: size of the scanning window
    #step_size: step size for scanning windows
    #min_overlap: minimum overlap required with a peak to consider it a modification site
   
    chrom, chrom_size, peaks, window_size, step_size, min_overlap = args
    
    modification_sites = []
    total_windows = 0
    
    #goes up to chrom_size -window_size + 1 to make sure we do not go out of bounds
    for window_start in range(0, chrom_size - window_size + 1, step_size):
        window_end = window_start + window_size
        total_windows += 1
        
        # check for sufficient overlap with ANY peak
        has_sufficient_overlap = False
        
        for peak_start, peak_end in peaks:
            if peak_end <= window_start:
                continue
            if peak_start >= window_end:
                break
            
            # calculate overlap
            overlap_start = max(window_start, peak_start)
            overlap_end = min(window_end, peak_end)
            overlap_length = overlap_end - overlap_start
            
            if overlap_length >= min_overlap:
                has_sufficient_overlap = True
                break
        
        if has_sufficient_overlap:
            modification_sites.append((chrom, window_start, window_end))
    
    return modification_sites, total_windows


def scan_genome_for_modification_sites(epigenome_id, marker, all_peaks=None, apply_threshold=True):
    
    start_time = log_progress(f"Scanning {epigenome_id}-{marker} for modification sites...")
    
    # use preloaded peaks if available, otherwise load individually
    if all_peaks and marker in all_peaks:
        peaks_by_chrom = all_peaks[marker]
        total_peaks = sum(len(peaks) for peaks in peaks_by_chrom.values())
        log_progress(f"Using preloaded {total_peaks:,} peaks from {len(peaks_by_chrom)} chromosomes")
    else:
        
        peaks_file = config.get_chipseq_path(epigenome_id, marker)
        if not os.path.exists(peaks_file):
            log_progress(f"Error: {peaks_file} not found")
            return []
        
        peaks_by_chrom = defaultdict(list)
        total_peaks = 0
        
        with open(peaks_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                cols = line.split('\t')
                if len(cols) < 3:
                    continue
                    
                try:
                    chrom, start, end = cols[0], int(cols[1]), int(cols[2])
                    
                    if chrom in ['chrX', 'chrY']:
                       continue
                    
                    if config.TEST_MODE and chrom != config.TEST_CHROMOSOME:
                        continue
                    
                    if start >= end or start < 0:
                        continue
                    
                    chrom_sizes = config.get_chrom_sizes()
                    if chrom not in chrom_sizes:
                        continue
                    
                    if end > chrom_sizes[chrom]:
                        end = chrom_sizes[chrom]
                    
                    if start < chrom_sizes[chrom]:  
                        peaks_by_chrom[chrom].append((start, end))
                        total_peaks += 1
                        
                except (ValueError, IndexError):
                    continue
        
        
        for chrom in peaks_by_chrom:
            peaks_by_chrom[chrom].sort()
        
        log_progress(f"Loaded {total_peaks:,} peaks from {len(peaks_by_chrom)} chromosomes")
    
    # Use cached chromosome sizes
    chrom_sizes = config.get_chrom_sizes()
    
    
    process_args = []
    for chrom in sorted(chrom_sizes.keys()):
        if chrom in peaks_by_chrom:
            #processing paramters, chromosome name, chromosome size, peaks for this chromosome, window size, step size, minimum overlap
            args = (chrom, chrom_sizes[chrom], peaks_by_chrom[chrom], 
                   config.WINDOW_SIZE, config.STEP_SIZE, config.MIN_OVERLAP)
            process_args.append(args)
    
  
    if len(process_args) > 1 and config.USE_MULTIPROCESSING:
        with mp.Pool(min(config.N_PROCESSES, len(process_args))) as pool:
            #different processes will handle different chromsomes
            #call to earlier scan_chromosome_parallel function - scanning each chromosome in parallel
            chrom_results = pool.map(scan_chromosome_parallel, process_args)
    else:
        
        chrom_results = [scan_chromosome_parallel(args) for args in process_args]
    
    
    modification_sites = [] #will store all valid genomic regions that overlap histone peaks
    total_windows = 0
    for chrom_sites, chrom_windows in chrom_results: #chrom results is a list of tuples (modification_sites (list), total_windows)
        modification_sites.extend(chrom_sites)
        total_windows += chrom_windows
        
        
        if chrom_sites:
            chrom = chrom_sites[0][0] if chrom_sites else "unknown"
            log_progress(f"  {chrom}: {len(chrom_sites):,} modification sites") #this is just to report progress tbh
    
    sites_count = len(modification_sites)
    log_progress(f"Found {sites_count:,} modification sites from {total_windows:,} windows scanned", start_time)
    
    
    if apply_threshold and sites_count < config.MIN_SITES_THRESHOLD and not config.TEST_MODE:
        log_progress(f"WARNING: Only {sites_count:,} sites found for {epigenome_id}-{marker}. "
                    f"Paper discards epigenomes with <{config.MIN_SITES_THRESHOLD:,} sites per marker.")
        return []
    
    return modification_sites #returns a list of valid genomic regions for downstream proessing


#checks if there are missing files for a given epigenome
def validate_epigenome_files(epigenome_id):
   
    missing_files = []
    
    
    for marker in config.ALL_MARKERS:
        chip_file = config.get_chipseq_path(epigenome_id, marker)
        if not os.path.exists(chip_file):
            missing_files.append(f"{marker} ChIP-seq")
    
   
    dnase_file = config.get_dnase_path(epigenome_id)
    if not os.path.exists(dnase_file):
        missing_files.append("DNase-seq")
    
    return len(missing_files) == 0, missing_files

#makes sure dataset does not already exist
def check_dataset_exists(epigenome_id, target_marker):
    
    output_path = config.get_output_path(epigenome_id, target_marker)
    return os.path.exists(output_path)

#makes sure that dataset is valid - we need certain keys
def validate_dataset_integrity(output_path):
    try:
        data = np.load(output_path, allow_pickle=True)
        required_keys = ['sequences', 'openness', 'labels', 'keys', 'metadata']
        
        for key in required_keys:
            if key not in data:
                return False, f"Missing key: {key}"
        
        
        n_samples = len(data['sequences'])
        if (len(data['openness']) != n_samples or 
            len(data['labels']) != n_samples or 
            len(data['keys']) != n_samples):
            return False, "Inconsistent sample counts across arrays"
        
       
        try:
            metadata = json.loads(str(data['metadata']))
            if 'epigenome_id' not in metadata or 'target_marker' not in metadata:
                return False, "Missing required metadata"
        except:
            return False, "Invalid metadata format"
        
        return True, "Dataset is valid"
        
    except Exception as e:
        return False, f"Error loading dataset: {str(e)}"


def load_all_histone_markers_for_epigenome(epigenome_id, target_marker):
    
    start_time = log_progress(f"Processing all histone markers for {epigenome_id}...")
    
    #just making sure again that the files exist
    files_valid, missing_files = validate_epigenome_files(epigenome_id)
    if not files_valid:
        raise FileNotFoundError(f"Missing files for {epigenome_id}: {missing_files}")
    
    #returns {marker: {chrom: [(start, end), ...]}}
    all_peaks = load_all_peaks_at_once(epigenome_id)
    
    all_marker_sites = {} # stores modification sites for each marker
    marker_stats = {} #stores site counts for logging -for myself
    
    #loops through all 7 markers and finds 200 bp windows that overlap significantly with this markers peaks
    for marker in config.ALL_MARKERS:
        marker_sites = scan_genome_for_modification_sites(
            epigenome_id, marker, all_peaks=all_peaks, apply_threshold=True
        )
        
        if not marker_sites and not config.TEST_MODE:
            log_progress(f"ERROR: {epigenome_id}-{marker} has insufficient sites, skipping epigenome")
            return None, None, None
        
        marker_sites_set = set(marker_sites) #removes duplicate sites
        all_marker_sites[marker] = marker_sites_set #save sites for this marker in main dic
        marker_stats[marker] = len(marker_sites_set) #for logging just counting sites
        
        log_progress(f"  {marker}: {len(marker_sites_set):,} sites")
    
    # Extract target and non-target sites
    if target_marker not in all_marker_sites:
        raise ValueError(f"Target marker {target_marker} not found in processed markers")
    
    target_sites = all_marker_sites[target_marker]
    
    # Combine other markers
    other_markers_sites = set()
    for marker, sites in all_marker_sites.items():
        if marker != target_marker:
            other_markers_sites.update(sites)
    
    # DeepHistone negative strategy: other markers - target marker
    #pretty much a negative set of OTHER sites that are not the target site
    negative_sites = other_markers_sites - target_sites
    
    
    all_sites_union = set()
    for sites in all_marker_sites.values():
        all_sites_union.update(sites)
    
    log_progress(f"Site statistics for {epigenome_id}:", start_time)
    log_progress(f"  Target ({target_marker}): {len(target_sites):,} sites")
    log_progress(f"  Other markers combined: {len(other_markers_sites):,} sites")
    log_progress(f"  Negatives (other - target): {len(negative_sites):,} sites")
    log_progress(f"  Total unique sites: {len(all_sites_union):,} sites")
    
    return list(target_sites), list(negative_sites), all_marker_sites

#are expanding to 1000 bp for the cnn -- might improve on this later
def expand_regions_to_1000bp(regions_200bp):

    start_time = log_progress(f"Expanding {len(regions_200bp):,} regions from 200bp to 1000bp...")
    
    chrom_sizes = load_chromosome_sizes()
    expanded_regions = []
    filtered_count = 0
    
    for chrom, start_200, end_200 in regions_200bp:
        center = (start_200 + end_200) // 2
        
        # Create 1000bp window centered on this position
        half_final = config.FINAL_WINDOW_SIZE // 2
        start_1000 = center - half_final
        end_1000 = center + half_final
        
        # Boundary checking
        if start_1000 < 0:
            start_1000 = 0
            end_1000 = config.FINAL_WINDOW_SIZE
        
        if chrom in chrom_sizes:
            chrom_size = chrom_sizes[chrom]
            if end_1000 > chrom_size:
                end_1000 = chrom_size
                start_1000 = max(0, chrom_size - config.FINAL_WINDOW_SIZE)
            
            if end_1000 - start_1000 >= config.FINAL_WINDOW_SIZE:
                end_1000 = start_1000 + config.FINAL_WINDOW_SIZE
                expanded_regions.append((chrom, start_1000, end_1000))
            else:
                filtered_count += 1
        else:
            if end_1000 - start_1000 >= config.FINAL_WINDOW_SIZE:
                expanded_regions.append((chrom, start_1000, start_1000 + config.FINAL_WINDOW_SIZE))
            else:
                filtered_count += 1
    
    if filtered_count > 0:
        log_progress(f"Filtered out {filtered_count:,} regions that couldn't form full 1000bp windows")
    
    log_progress(f"Successfully expanded {len(expanded_regions):,} regions", start_time)
    return expanded_regions


def extract_sequences(regions):
    
    start_time = log_progress(f"Extracting DNA sequences for {len(regions):,} regions...")
    
    try:
        genome = config.get_genome()  # Use cached genome
    except Exception as e:
        raise FileNotFoundError(f"Cannot load genome FASTA: {config.FASTA_PATH}. Error: {e}")
    
    sequences = []
    invalid_count = 0
    
   
    chunk_size = 5000
    for i in range(0, len(regions), chunk_size):
        chunk = regions[i:i + chunk_size]
        
        for chrom, region_start, region_end in chunk:
            expected_length = region_end - region_start
            
            try:
                #ex genome['chr1'][600:1600] returns 1000bp of DNA from chromosome 1
                seq = genome[chrom][region_start:region_end].seq.upper() #using pyfaidx to get DNA from genome FASTA, standardizing to uppercase
                
                #making sure it is the correct length if not pad with Ns or cut
                if len(seq) != expected_length:
                    if len(seq) < expected_length:
                        seq = seq.ljust(expected_length, 'N')
                    else:
                        seq = seq[:expected_length]
                
                n_count = seq.count('N')
                n_fraction = n_count / len(seq)
                
                #only let 10% of the sequence be Ns
                if n_fraction > config.MAX_N_FRACTION:
                    invalid_count += 1
                    seq = 'N' * expected_length
                
                sequences.append(seq)
                
            except Exception as e:
                log_progress(f"Warning: Could not extract sequence for {chrom}:{region_start}-{region_end}: {e}")
                sequences.append('N' * expected_length)
                invalid_count += 1
        
        # progress update
        if i % (chunk_size * 4) == 0 and i > 0:
            log_progress(f"Extracted {i + len(chunk):,}/{len(regions):,} sequences...")
    
    if invalid_count > 0:
        log_progress(f"Warning: {invalid_count:,} sequences had quality issues")
    
    log_progress(f"Extracted {len(sequences):,} sequences", start_time)
    return sequences


def extract_dnase_openness_scores(epigenome_id, regions):

    start_time = log_progress(f"Extracting DNase openness scores for {len(regions):,} regions...")
    
    dnase_file = config.get_dnase_path(epigenome_id) #get file path
    dnase_peaks_by_chrom = defaultdict(list)
    
    if not os.path.exists(dnase_file):
        log_progress(f"Warning: DNase file {dnase_file} not found, using zero openness scores")
        return [np.zeros(config.FINAL_WINDOW_SIZE, dtype=np.float32) for _ in regions] #if there is no DNase file, return zero scores for all regions -- ie closed chromatin
    
    # Parse DNase peaks
    total_dnase_peaks = 0
    with open(dnase_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            cols = line.split('\t')
            if len(cols) < 7:
                continue
                
            try:
                # col[0] is chromosome name, col[1] is start position, col[2] is end position
                chrom, start, end = cols[0], int(cols[1]), int(cols[2])
                 
                 #SEX CHROMOSOME
                if chrom in ['chrX', 'chrY']:
                  continue
                
                if config.TEST_MODE and chrom != config.TEST_CHROMOSOME:
                    continue
                #using signal value from column 6 as fold enrichment
                fold_enrichment = 1.0
                try:
                    if len(cols) > 6:
                        fold_enrichment = float(cols[6])
                    else: 
                        fold_enrichment = 1.0
                except(ValueError, IndexError):
                     fold_enrichment = 1.0 
                
                fold_enrichment = max(0.0, fold_enrichment)
                dnase_peaks_by_chrom[chrom].append((start, end, fold_enrichment))
                total_dnase_peaks += 1
                
            except (ValueError, IndexError) as e:
                continue
    
    
    for chrom in dnase_peaks_by_chrom:
        dnase_peaks_by_chrom[chrom].sort() #order peaks by start coordinate within each chromosome so we will have a chromosmome with a tuple of peaks (start, end, fold_enrichment)
    
    log_progress(f"Loaded {total_dnase_peaks:,} DNase peaks")
    
    
    openness_scores = []
    
    # VALID_CHROMOSOMES = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}

    for region_idx, (chrom, region_start, region_end) in enumerate(tqdm(regions, desc="Extracting openness")):
        
        # if chrom not in VALID_CHROMOSOMES:
        #     continue
        
        region_length = region_end - region_start
        openness = np.zeros(region_length, dtype=np.float32)

        # if np.all(openness == 0):
        #     continue # Skip if no DNase peaks for this region
        
        if chrom in dnase_peaks_by_chrom:
            for peak_start, peak_end, fold_enrichment in dnase_peaks_by_chrom[chrom]:
                if peak_end <= region_start or peak_start >= region_end:
                    continue
                
                overlap_start = max(region_start, peak_start)
                overlap_end = min(region_end, peak_end)
                
                if overlap_start < overlap_end:
                    start_idx = overlap_start - region_start
                    end_idx = overlap_end - region_start
                    openness[start_idx:end_idx] = fold_enrichment #looking at positions and openness at those positions will be the fold_enrichment value
        
        openness_scores.append(openness)
    
    log_progress(f"Extracted openness scores for {len(regions):,} regions", start_time)
    return openness_scores


def create_natural_imbalanced_dataset(pos_sequences, pos_openness, neg_sequences, neg_openness):
    start_time = log_progress("Creating dataset with natural class distribution...")
    
    pos_count = len(pos_sequences)
    neg_count = len(neg_sequences)
    
    if pos_count == 0:
        raise ValueError("No positive samples available")
    if neg_count == 0:
        raise ValueError("No negative samples available")
    
    all_sequences = pos_sequences + neg_sequences
    all_openness = pos_openness + neg_openness
    all_labels = np.array([1] * pos_count + [0] * neg_count, dtype=np.int32)
    
    natural_ratio = neg_count / pos_count
    log_progress(f"Created dataset: {pos_count:,} pos + {neg_count:,} neg = {len(all_sequences):,} total", start_time)
    log_progress(f"Natural class distribution ratio: {natural_ratio:.1f}:1 (negative:positive)")
    
    #example output
    # Sequence: "ATCGATCG..." (1000bp DNA)
    # Accessibility: [2.1, 0, 3.4, 0, 1.8, ...] (1000 values)
    # Label: 1 (positive for target marker)
    return all_sequences, all_openness, all_labels #these are the inputs to the machine learning model sequence is the sequence openness is the openness score and labels are the class labels (1 for positive, 0 for negative)


def save_dataset_with_metadata(output_path, sequences, openness, labels, epigenome_id, target_marker, 
                             genomic_keys=None, metadata=None):
   
    start_time = log_progress("Saving dataset...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    max_len = max(len(seq) for seq in sequences) if sequences else 0
    sequences_array = np.array([list(seq.ljust(max_len, 'N')) for seq in sequences], dtype='U1')
    openness_array = np.array(openness, dtype=np.float32)
    
    if genomic_keys: #should have genomic keys if we are using real genomic regions
        keys_array = np.array(genomic_keys, dtype='U30')
    else: #generate fake ones if necessary
        n_samples = len(sequences)
        chr22_start = 16000000
        keys = [f"chr22:{chr22_start + i*1200}-{chr22_start + i*1200 + 1000}" for i in range(n_samples)]
        keys_array = np.array(keys, dtype='U30')
    
    save_metadata = {
        'epigenome_id': epigenome_id,
        'target_marker': target_marker,
        'window_size': config.WINDOW_SIZE,
        'final_window_size': config.FINAL_WINDOW_SIZE,
        'step_size': config.STEP_SIZE,
        'min_overlap': config.MIN_OVERLAP,
        'min_sites_threshold': config.MIN_SITES_THRESHOLD,
        'all_markers': config.ALL_MARKERS,
        'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_mode': config.TEST_MODE
    }
    
    if metadata:
        save_metadata.update(metadata)
    #save to compressed archive
    np.savez_compressed(
        output_path,
        sequences=sequences_array,
        openness=openness_array,
        labels=labels,
        keys=keys_array,
        metadata=json.dumps(save_metadata)
    )
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    pos_count = label_counts.get(1, 0)
    neg_count = label_counts.get(0, 0)
    
    file_size_mb = os.path.getsize(output_path) / (1024*1024)
    
    print(f"\n{'='*50}")
    print(f"DATASET SAVED: {os.path.basename(output_path)}")
    print(f"{'='*50}")
    print(f"Epigenome: {epigenome_id}")
    print(f"Target marker: {target_marker}")
    print(f"Total samples: {len(sequences):,}")
    print(f"Positive samples: {pos_count:,} ({pos_count/len(sequences)*100:.1f}%)")
    print(f"Negative samples: {neg_count:,} ({neg_count/len(sequences)*100:.1f}%)")
    if pos_count > 0 and neg_count > 0:
        print(f"Class ratio (neg:pos): {neg_count/pos_count:.1f}:1")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"File path: {output_path}")
    
    log_progress(f"Dataset saved successfully", start_time)
    return output_path

#DEBUG - find overlap and then expand (was getting super low AUPRC scores and it because DNAse alignment scores suck)
def expand_dnase_scores_to_1000bp(openness_200bp):
    expanded_scores = []
    
    for scores_200 in openness_200bp:
        
        scores_1000 = np.zeros(1000, dtype=np.float32)
        
        
        start_idx = (1000 - 200) // 2  
        end_idx = start_idx + 200      
        
        scores_1000[start_idx:end_idx] = scores_200
        expanded_scores.append(scores_1000)
    
    return expanded_scores

#jumbo function that runs the entire pipeline for a single epigenome and marker combination
def run_single_combination(epigenome_id, target_marker, logger=None):
    
    overall_start = time.time()

    try:
        if logger:
            logger.info(f"Starting {epigenome_id}-{target_marker}")

        print(f"\n{'='*60}")
        print(f"Processing: {epigenome_id} - {target_marker}")
        print(f"{'='*60}")

        #if output file exists skip processing it or reprocess if not valid
        if getattr(config, "SKIP_EXISTING", False) and check_dataset_exists(epigenome_id, target_marker):
            output_path = config.get_output_path(epigenome_id, target_marker)
            is_valid, msg = validate_dataset_integrity(output_path)
            if is_valid:
                print(f"Dataset already exists and is valid: {output_path}")
                if logger:
                    logger.info(f"Skipped {epigenome_id}-{target_marker} - already exists")
                return output_path, True
            else:
                print(f"Existing dataset is invalid ({msg}), reprocessing...")
                if logger:
                    logger.warning(f"Invalid dataset for {epigenome_id}-{target_marker}: {msg}")

        
        if target_marker not in config.ALL_MARKERS:
            raise ValueError(f"Target marker '{target_marker}' not in valid marker list: {config.ALL_MARKERS}")

        # Load all peaks for the epigenome
        target_sites_200bp, negative_sites_200bp, all_marker_sites = load_all_histone_markers_for_epigenome(
            epigenome_id, target_marker
        )

        if target_sites_200bp is None:
            raise ValueError(f"Failed to process {epigenome_id} - missing sufficient data for at least one marker")

        #regions iwth target marker peaks and negative regions with other markers expaned to 1000bp
        # FIXED: Extract DNase scores from 200bp regions FIRST (better overlap)
        log_progress("Extracting DNase scores from 200bp regions...")
        pos_openness_200bp = extract_dnase_openness_scores(epigenome_id, target_sites_200bp)
        neg_openness_200bp = extract_dnase_openness_scores(epigenome_id, negative_sites_200bp)

        # expand regions to 1000bp for sequence extraction
        target_sites_1000bp = expand_regions_to_1000bp(target_sites_200bp)
        negative_sites_1000bp = expand_regions_to_1000bp(negative_sites_200bp)

        if len(target_sites_1000bp) == 0:
            raise ValueError("No valid positive regions after expansion to 1000bp")
        if len(negative_sites_1000bp) == 0:
            raise ValueError("No valid negative regions after expansion to 1000bp")

        # extract DNA sequences from 1000bp regions
        pos_sequences = extract_sequences(target_sites_1000bp)
        neg_sequences = extract_sequences(negative_sites_1000bp)

        # expand DNase scores to match 1000bp regions
        log_progress("Expanding DNase scores to 1000bp...")
        pos_openness = expand_dnase_scores_to_1000bp(pos_openness_200bp)
        neg_openness = expand_dnase_scores_to_1000bp(neg_openness_200bp)

        # merge the two
        sequences, openness, labels = create_natural_imbalanced_dataset(
            pos_sequences, pos_openness, neg_sequences, neg_openness
        )

    
        all_regions_1000bp = target_sites_1000bp + negative_sites_1000bp
        genomic_keys = [f"{chrom}:{start}-{end}" for chrom, start, end in all_regions_1000bp]

        np.random.seed(config.RANDOM_SEED)
        indices = np.random.permutation(len(sequences))

        sequences = [sequences[i] for i in indices]
        openness = [openness[i] for i in indices]
        labels = labels[indices]
        genomic_keys = [genomic_keys[i] for i in indices]


        metadata = {
            'uses_natural_distribution': True,
            'paper_methodology': 'DeepHistone',
            'original_positive_count': len(pos_sequences),
            'original_negative_count': len(neg_sequences),
            'final_dataset_size': len(sequences),
            'natural_ratio': len(neg_sequences) / len(pos_sequences) if len(pos_sequences) > 0 else 0,
            'optimized_version': True,
            'optimization_features': [
                'cached_genome_loading',
                'parallel_chromosome_processing', 
                'bulk_peak_loading',
                'chunked_sequence_extraction'
            ]
        }

        output_path = config.get_output_path(epigenome_id, target_marker)
        save_dataset_with_metadata(
            output_path, sequences, openness, labels,
            epigenome_id, target_marker,
            genomic_keys=genomic_keys,
            metadata=metadata
        )

        duration = time.time() - overall_start
        log_progress(f"SUCCESS: Finished {epigenome_id}-{target_marker} in {duration:.2f} seconds")
        if logger:
            logger.info(f"SUCCESS: Completed {epigenome_id}-{target_marker} in {duration:.2f} seconds")

        return output_path, True

    except Exception as e:
        error_msg = f"Error processing {epigenome_id}-{target_marker}: {e}"
        print(error_msg)
        print(traceback.format_exc())
        if logger:
            logger.error(error_msg)
            logger.error(traceback.format_exc())
        return None, False


def run_batch_processing(combinations=None, logger=None):
    if logger is None:
        logger = setup_logging()
    
    if combinations is None:
        combinations = config.get_all_combinations()
    
    logger.info(f"Starting batch processing of {len(combinations)} combinations")
    
    successful = []
    failed = []
    skipped = []
    
    start_time = time.time()
    
    for i, (epigenome_id, target_marker) in enumerate(combinations, 1):
        try:
            logger.info(f"Processing {i}/{len(combinations)}: {epigenome_id}-{target_marker}")
            
            
            if config.SKIP_EXISTING and check_dataset_exists(epigenome_id, target_marker):
                output_path = config.get_output_path(epigenome_id, target_marker)
                is_valid, msg = validate_dataset_integrity(output_path)
                if is_valid:
                    logger.info(f"Skipping {epigenome_id}-{target_marker} - already exists and valid")
                    skipped.append((epigenome_id, target_marker))
                    continue
                else:
                    logger.warning(f"Existing dataset invalid for {epigenome_id}-{target_marker}: {msg}")
            
            # combo
            output_path, success = run_single_combination(epigenome_id, target_marker, logger)
            
            if success:
                successful.append((epigenome_id, target_marker, output_path))
                logger.info(f"SUCCESS: {epigenome_id}-{target_marker}")
            else:
                failed.append((epigenome_id, target_marker, "Processing failed"))
                logger.error(f"FAILED: {epigenome_id}-{target_marker}")
                
                if not config.CONTINUE_ON_ERROR:
                    logger.error("Stopping batch processing due to error")
                    break
        
        except Exception as e:
            error_msg = f"Unexpected error with {epigenome_id}-{target_marker}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            failed.append((epigenome_id, target_marker, str(e)))
            
            if not config.CONTINUE_ON_ERROR:
                logger.error("Stopping batch processing due to unexpected error")
                break
        
        # progress
        elapsed = time.time() - start_time
        remaining = len(combinations) - i
        if i > 0:
            avg_time = elapsed / i
            eta = avg_time * remaining
            logger.info(f"Progress: {i}/{len(combinations)} completed. "
                       f"ETA: {eta/3600:.1f} hours. "
                       f"Success: {len(successful)}, Failed: {len(failed)}, Skipped: {len(skipped)}")
    
    # final summary
    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"BATCH PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Skipped: {len(skipped)}")
    logger.info(f"Total processed: {len(successful) + len(failed) + len(skipped)}")
    
    if successful:
        logger.info("\nSuccessful datasets:")
        for epigenome, marker, path in successful:
            logger.info(f"  {epigenome}-{marker}: {path}")
    
    if failed:
        logger.info("\nFailed datasets:")
        for epigenome, marker, error in failed:
            logger.info(f"  {epigenome}-{marker}: {error}")
    
    return successful, failed, skipped


def check_current_chromosomes():
    """Check what chromosomes are currently being processed"""
    print("\n" + "="*50)
    print("CURRENT CHROMOSOME STATUS CHECK")
    print("="*50)
    
    # Check chromosome sizes
    chrom_sizes = config.get_chrom_sizes()
    print(f"Chromosomes in chromosome sizes: {sorted(chrom_sizes.keys())}")
    print(f"Number of chromosomes: {len(chrom_sizes)}")
    
    # Check for sex chromosomes
    sex_chroms = [c for c in chrom_sizes.keys() if c in ['chrX', 'chrY']]
    if sex_chroms:
        print(f"❌ Sex chromosomes found: {sex_chroms}")
    else:
        print("✅ No sex chromosomes in chromosome sizes")
    
    # Test peak loading for one epigenome
    try:
        print("\nTesting peak loading...")
        peaks = load_all_peaks_at_once("E003")
        
        all_chroms_in_peaks = set()
        for marker in peaks:
            all_chroms_in_peaks.update(peaks[marker].keys())
        
        print(f"Chromosomes found in E003 peaks: {sorted(all_chroms_in_peaks)}")
        
        sex_in_peaks = [c for c in all_chroms_in_peaks if c in ['chrX', 'chrY']]
        if sex_in_peaks:
            print(f"❌ Sex chromosomes in peaks: {sex_in_peaks}")
        else:
            print("✅ No sex chromosomes in peak data")
            
    except Exception as e:
        print(f"Could not test peaks: {e}")
    
    print("="*50)



if __name__ == "__main__":
    check_current_chromosomes()
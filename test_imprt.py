#!/usr/bin/env python3
"""
Test your imports and basic functionality
"""

try:
    from deephistone_pipeline_all import run_single_combination, setup_logging, config
    print("✅ SUCCESS: All imports working")
    
    # Test config
    print(f"✅ Valid epigenomes: {config.VALID_EPIGENOMES[:3]}...")
    print(f"✅ All markers: {config.ALL_MARKERS}")
    print(f"✅ Test mode: {config.TEST_MODE}")
    
    # Test if E003 is in valid epigenomes
    if 'E003' in config.VALID_EPIGENOMES:
        print("✅ E003 is in valid epigenomes")
    else:
        print("❌ WARNING: E003 not in valid epigenomes")
    
    # Test your target markers
    target_markers = ["H3K36me3", "H3K27me3", "H3K9me3", "H3K27ac", "H3K9ac"]
    missing_markers = [m for m in target_markers if m not in config.ALL_MARKERS]
    
    if not missing_markers:
        print("✅ All target markers are valid")
    else:
        print(f"❌ WARNING: Missing markers: {missing_markers}")
    
    print("\n🎉 Import test successful - ready to proceed!")
    
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    print("Check that deephistone_pipeline_all.py is in your current directory")
    
except Exception as e:
    print(f"❌ OTHER ERROR: {e}")

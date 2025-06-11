#!/usr/bin/env python3
"""
Dependency checker for DeepHistone 5-fold cross-validation
"""

import sys
import importlib
import subprocess

def check_module(module_name, package_name=None):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name}")
        return True
    except ImportError as e:
        package = package_name or module_name
        print(f"✗ {module_name} - Missing! Install with: pip install {package}")
        return False

def check_pytorch_functionality():
    """Check PyTorch specific functionality"""
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        # Test basic operations
        x = torch.randn(2, 3)
        print(f"✓ Basic tensor operations work")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available (CPU-only mode)")
        
        # Test optimizer creation (this was failing)
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        print("✓ Optimizer creation works")
        
        return True
    except Exception as e:
        print(f"✗ PyTorch functionality test failed: {e}")
        return False

def check_data_files():
    """Check if required data files exist"""
    import os
    files_to_check = [
        "data/converted/mini_merged.npz",
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✓ {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {file_path} - File not found!")

def check_custom_modules():
    """Check custom project modules"""
    import os
    sys.path.append('.')  # Add current directory to path
    
    modules_to_check = [
        ("model", "DeepHistone"),
        ("utils", "metrics, model_train, model_eval, model_predict")
    ]
    
    for module_name, components in modules_to_check:
        try:
            module = importlib.import_module(module_name)
            print(f"✓ {module_name}.py found")
            
            # Check if specific classes/functions exist
            if module_name == "model":
                if hasattr(module, 'DeepHistone'):
                    print(f"  ✓ DeepHistone class available")
                else:
                    print(f"  ✗ DeepHistone class not found!")
            elif module_name == "utils":
                utils_funcs = ['metrics', 'model_train', 'model_eval', 'model_predict']
                for func in utils_funcs:
                    if hasattr(module, func):
                        print(f"  ✓ {func} function available")
                    else:
                        print(f"  ✗ {func} function not found!")
        except ImportError:
            print(f"✗ {module_name}.py - Module not found!")

def main():
    print("=" * 60)
    print("DEEPHISTONE DEPENDENCY CHECKER")
    print("=" * 60)
    
    print("\n1. Checking Python standard library modules:")
    print("-" * 40)
    all_good = True
    
    # Basic Python modules
    standard_modules = [
        "os", "sys", "copy", "collections"
    ]
    
    for module in standard_modules:
        all_good &= check_module(module)
    
    print("\n2. Checking scientific computing packages:")
    print("-" * 40)
    
    # Scientific packages with specific package names for installation
    sci_packages = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("sklearn", "scikit-learn"),
        ("sympy", "sympy")
    ]
    
    for module, package in sci_packages:
        all_good &= check_module(module, package)
    
    print("\n3. Checking PyTorch:")
    print("-" * 40)
    torch_ok = check_module("torch", "torch")
    if torch_ok:
        pytorch_functional = check_pytorch_functionality()
        all_good &= pytorch_functional
    else:
        all_good = False
    
    print("\n4. Checking custom project modules:")
    print("-" * 40)
    check_custom_modules()
    
    print("\n5. Checking data files:")
    print("-" * 40)
    check_data_files()
    
    print("\n6. System information:")
    print("-" * 40)
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 ALL CORE DEPENDENCIES SATISFIED!")
        print("You should be able to run the 5-fold cross-validation.")
    else:
        print("❌ SOME DEPENDENCIES ARE MISSING!")
        print("Please install missing packages before running the training.")
    print("=" * 60)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
GraphLAMA End-to-End Runner

This is the main entry point for running GraphLAMA end-to-end.
It provides a menu-driven interface for different execution modes.
"""

import sys
import os
import subprocess
from pathlib import Path

def print_header():
    print("\n" + "=" * 70)
    print("  GraphLAMA - Graph Language Model Adaptation")
    print("  End-to-End Runner")
    print("=" * 70)

def print_menu():
    print("\nSelect an option:")
    print("  1. Quick Demo (minimal dependencies)")
    print("  2. Cora Dataset Example (data preparation)")
    print("  3. Check Dependencies")
    print("  4. Help & Documentation")
    print("  0. Exit")
    print()

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\nChecking dependencies...")
    print("-" * 70)
    
    deps = {
        'torch': 'PyTorch',
        'torch_geometric': 'PyTorch Geometric',
        'transformers': 'Hugging Face Transformers',
        'numpy': 'NumPy',
    }
    
    optional_deps = {
        'torch_scatter': 'PyG Scatter (optional)',
        'torch_sparse': 'PyG Sparse (optional)',
        'sentencepiece': 'SentencePiece (for LLMs)',
        'fschat': 'FastChat (for Vicuna)',
    }
    
    all_ok = True
    
    print("\nRequired Dependencies:")
    for module, name in deps.items():
        try:
            __import__(module)
            version = __import__(module).__version__ if hasattr(__import__(module), '__version__') else 'unknown'
            print(f"  ✓ {name}: {version}")
        except ImportError:
            print(f"  ✗ {name}: NOT INSTALLED")
            all_ok = False
    
    print("\nOptional Dependencies:")
    for module, name in optional_deps.items():
        try:
            __import__(module)
            version = __import__(module).__version__ if hasattr(__import__(module), '__version__') else 'installed'
            print(f"  ✓ {name}: {version}")
        except ImportError:
            print(f"  - {name}: not installed")
    
    print("-" * 70)
    
    if not all_ok:
        print("\n⚠ Some required dependencies are missing!")
        print("\nTo install minimal dependencies:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print("  pip install torch_geometric transformers")
        print("\nFor full functionality, see RUN_END_TO_END.md")
    else:
        print("\n✓ All required dependencies are installed!")
    
    return all_ok

def run_quick_demo():
    """Run the quick demo with synthetic data."""
    print("\nRunning Quick Demo...")
    print("-" * 70)
    
    script = Path("simple_demo.py")
    if not script.exists():
        print(f"Error: {script} not found!")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(script)], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\nError running demo: {e}")
        return False

def run_cora_example():
    """Run the Cora dataset example."""
    print("\nRunning Cora Dataset Example...")
    print("-" * 70)
    
    script = Path("run_cora_example.py")
    if not script.exists():
        print(f"Error: {script} not found!")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(script)], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\nError running Cora example: {e}")
        return False

def show_help():
    """Display help information."""
    print("\nGraphLAMA End-to-End Help")
    print("-" * 70)
    print("\nThis runner provides easy access to GraphLAMA examples and demos.\n")
    
    print("Option 1: Quick Demo")
    print("  - Runs a minimal demo with synthetic data")
    print("  - Requires: PyTorch, PyTorch Geometric")
    print("  - Output: demo_output.json")
    
    print("\nOption 2: Cora Dataset Example")
    print("  - Downloads and processes Cora citation network")
    print("  - Generates training/test conversation data")
    print("  - Output: data/cora_processed/*.json")
    
    print("\nOption 3: Check Dependencies")
    print("  - Verifies installed packages")
    print("  - Shows version information")
    print("  - Provides installation instructions if needed")
    
    print("\nFor detailed documentation, see:")
    print("  - RUN_END_TO_END.md - Step-by-step guide")
    print("  - README.md - Full project documentation")
    
    print("\nFor full training pipeline:")
    print("  1. Set up environment (see RUN_END_TO_END.md)")
    print("  2. Run data preparation (Option 2)")
    print("  3. Configure training scripts in scripts/tune_script/")
    print("  4. Run training: bash scripts/tune_script/stage1.sh")
    print("  5. Evaluate: bash scripts/tune_script/SFTonGFM_eval.sh")
    
    print("-" * 70)

def main():
    """Main entry point."""
    print_header()
    
    while True:
        print_menu()
        
        try:
            choice = input("Enter your choice (0-4): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
            break
        
        if choice == '0':
            print("\nGoodbye!")
            break
        elif choice == '1':
            if not check_dependencies():
                print("\nPlease install required dependencies first.")
                input("\nPress Enter to continue...")
                continue
            run_quick_demo()
            input("\nPress Enter to continue...")
        elif choice == '2':
            if not check_dependencies():
                print("\nPlease install required dependencies first.")
                input("\nPress Enter to continue...")
                continue
            run_cora_example()
            input("\nPress Enter to continue...")
        elif choice == '3':
            check_dependencies()
            input("\nPress Enter to continue...")
        elif choice == '4':
            show_help()
            input("\nPress Enter to continue...")
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()

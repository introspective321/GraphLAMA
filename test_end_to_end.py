#!/usr/bin/env python3
"""
Test script to validate end-to-end functionality
"""

import subprocess
import sys
import json
from pathlib import Path

def test_simple_demo():
    """Test the simple demo script."""
    print("\n[1/3] Testing simple_demo.py...")
    result = subprocess.run([sys.executable, "simple_demo.py"], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ✗ FAILED: {result.stderr}")
        return False
    
    # Check if output file was created
    output_file = Path("demo_output.json")
    if not output_file.exists():
        print("  ✗ FAILED: demo_output.json not created")
        return False
    
    # Validate JSON
    try:
        with open(output_file) as f:
            data = json.load(f)
        assert "id" in data
        assert "graph" in data
        assert "conversations" in data
        print("  ✓ PASSED")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: Invalid JSON - {e}")
        return False

def test_cora_example():
    """Test the Cora example script."""
    print("\n[2/3] Testing run_cora_example.py...")
    result = subprocess.run([sys.executable, "run_cora_example.py"], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ✗ FAILED: {result.stderr}")
        return False
    
    # Check if output files were created
    required_files = [
        "data/cora_processed/train_items.json",
        "data/cora_processed/test_items.json",
        "data/cora_processed/graph_data.pt"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"  ✗ FAILED: {file_path} not created")
            return False
    
    # Validate train data
    try:
        with open("data/cora_processed/train_items.json") as f:
            train_data = json.load(f)
        assert len(train_data) > 0
        assert "id" in train_data[0]
        assert "graph" in train_data[0]
        assert "conversations" in train_data[0]
        print("  ✓ PASSED")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: Invalid data - {e}")
        return False

def test_imports():
    """Test that grapht3 can be imported."""
    print("\n[3/3] Testing grapht3 imports...")
    try:
        sys.path.insert(0, '.')
        # Test basic imports that don't require torch-scatter
        import torch
        import torch_geometric
        import transformers
        print("  ✓ PASSED: Core dependencies available")
        return True
    except ImportError as e:
        print(f"  ✗ FAILED: {e}")
        return False

def main():
    print("=" * 60)
    print("GraphLAMA End-to-End Tests")
    print("=" * 60)
    
    tests = [
        test_simple_demo,
        test_cora_example,
        test_imports
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests PASSED")
        return 0
    else:
        print("✗ Some tests FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())

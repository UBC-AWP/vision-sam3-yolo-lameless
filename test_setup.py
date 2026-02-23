#!/usr/bin/env python3
"""
Quick test script to verify setup
"""
import os
import sys
from pathlib import Path

def test_structure():
    """Test project structure"""
    print("Testing project structure...")
    
    required_dirs = [
        "services/video-ingestion",
        "services/video-preprocessing",
        "shared/config",
        "shared/utils",
        "data/videos",
        "data/processed"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - MISSING")
            all_exist = False
    
    return all_exist

def test_files():
    """Test required files"""
    print("\nTesting required files...")
    
    required_files = [
        "docker-compose.yml",
        "environment.yml",
        "shared/config/config.yaml",
        "services/video-ingestion/app/main.py",
        "services/video-preprocessing/app/main.py",
        "shared/utils/nats_client.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def test_config():
    """Test config file"""
    print("\nTesting configuration...")
    try:
        import yaml
        config_path = Path("shared/config/config.yaml")
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            print("✓ Config file is valid YAML")
            print(f"  - NATS URL: {config.get('nats', {}).get('url', 'N/A')}")
            return True
        else:
            print("✗ Config file not found")
            return False
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Lameness Detection System Setup")
    print("=" * 50)
    
    structure_ok = test_structure()
    files_ok = test_files()
    config_ok = test_config()
    
    print("\n" + "=" * 50)
    if structure_ok and files_ok and config_ok:
        print("✓ All tests passed! Setup looks good.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Please check the issues above.")
        sys.exit(1)


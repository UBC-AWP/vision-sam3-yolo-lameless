#!/usr/bin/env python3
"""
Test script for Docker services
"""
import requests
import time
import sys
from pathlib import Path

API_BASE = "http://localhost:8000"
FRONTEND_BASE = "http://localhost:3000"

def test_backend_health():
    """Test backend health endpoint"""
    print("Testing backend health...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Backend is healthy")
            return True
        else:
            print(f"✗ Backend returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Backend not reachable: {e}")
        return False

def test_backend_api():
    """Test backend API endpoints"""
    print("\nTesting backend API endpoints...")
    
    # Test root endpoint
    try:
        response = requests.get(f"{API_BASE}/", timeout=5)
        if response.status_code == 200:
            print("✓ Root endpoint works")
        else:
            print(f"✗ Root endpoint returned {response.status_code}")
    except Exception as e:
        print(f"✗ Root endpoint error: {e}")
    
    # Test videos list
    try:
        response = requests.get(f"{API_BASE}/api/videos", timeout=5)
        if response.status_code == 200:
            print("✓ Videos list endpoint works")
        else:
            print(f"✗ Videos list returned {response.status_code}")
    except Exception as e:
        print(f"✗ Videos list error: {e}")
    
    # Test training stats
    try:
        response = requests.get(f"{API_BASE}/api/training/stats", timeout=5)
        if response.status_code == 200:
            print("✓ Training stats endpoint works")
        else:
            print(f"✗ Training stats returned {response.status_code}")
    except Exception as e:
        print(f"✗ Training stats error: {e}")
    
    # Test model parameters
    try:
        response = requests.get(f"{API_BASE}/api/models/parameters", timeout=5)
        if response.status_code == 200:
            print("✓ Model parameters endpoint works")
        else:
            print(f"✗ Model parameters returned {response.status_code}")
    except Exception as e:
        print(f"✗ Model parameters error: {e}")

def test_frontend():
    """Test frontend accessibility"""
    print("\nTesting frontend...")
    try:
        response = requests.get(FRONTEND_BASE, timeout=5)
        if response.status_code == 200:
            print("✓ Frontend is accessible")
            return True
        else:
            print(f"✗ Frontend returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Frontend not reachable: {e}")
        return False

def wait_for_services(max_wait=60):
    """Wait for services to be ready"""
    print("Waiting for services to start...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{API_BASE}/health", timeout=2)
            if response.status_code == 200:
                print("✓ Services are ready!")
                return True
        except:
            pass
        time.sleep(2)
        print(".", end="", flush=True)
    
    print("\n✗ Services did not become ready in time")
    return False

if __name__ == "__main__":
    print("=" * 50)
    print("Docker Services Test")
    print("=" * 50)
    
    # Wait for services
    if not wait_for_services():
        print("\n⚠ Services may still be starting. Continuing with tests...")
    
    # Run tests
    backend_ok = test_backend_health()
    if backend_ok:
        test_backend_api()
    
    test_frontend()
    
    print("\n" + "=" * 50)
    if backend_ok:
        print("✓ Basic tests passed!")
        print(f"\nAccess points:")
        print(f"  - Frontend: {FRONTEND_BASE}")
        print(f"  - Backend API: {API_BASE}")
        print(f"  - API Docs: {API_BASE}/docs")
    else:
        print("✗ Some tests failed. Check service logs.")
    print("=" * 50)



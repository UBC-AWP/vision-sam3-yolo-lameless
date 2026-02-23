#!/usr/bin/env python3
"""
Test script for raw videos API endpoint.
Tests the /api/videos/storage/raw-videos endpoint with different scenarios.
"""
import os
import sys
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import httpx
import boto3

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

# S3 Configuration
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_VIDEOS_BUCKET = os.getenv("S3_VIDEOS_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("S3_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("S3_SECRET_ACCESS_KEY")

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "demo123")


def test_s3_connection():
    """Test S3 connection and list objects."""
    print("=" * 80)
    print("Testing S3 Connection")
    print("=" * 80)
    
    try:
        s3 = boto3.client(
            's3',
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        
        # List objects under Farm1/raw/
        prefix = "Farm1/raw/"
        print(f"\nListing objects with prefix: {prefix}")
        response = s3.list_objects_v2(Bucket=S3_VIDEOS_BUCKET, Prefix=prefix)
        
        if 'Contents' in response:
            print(f"Found {len(response['Contents'])} objects:")
            for obj in response['Contents']:
                print(f"  - {obj['Key']} ({obj['Size']} bytes)")
        else:
            print("No objects found")
        
        # List all tenants
        print("\nListing all top-level prefixes:")
        response = s3.list_objects_v2(Bucket=S3_VIDEOS_BUCKET, Delimiter='/')
        if 'CommonPrefixes' in response:
            for prefix_info in response['CommonPrefixes']:
                print(f"  - {prefix_info['Prefix']}")
        
        return True
    except Exception as e:
        print(f"Error testing S3: {e}")
        return False


async def get_auth_token():
    """Get authentication token."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_BASE_URL}/api/auth/login",
            json={
                "email": ADMIN_EMAIL,
                "password": ADMIN_PASSWORD
            }
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("access_token")
        else:
            print(f"Login failed: {response.status_code} - {response.text}")
            return None


async def test_api_endpoint(tenant_id=None):
    """Test the raw videos API endpoint."""
    print("=" * 80)
    print(f"Testing API Endpoint: /api/videos/storage/raw-videos")
    if tenant_id:
        print(f"With tenant_id filter: {tenant_id}")
    else:
        print("Without tenant_id filter (all tenants)")
    print("=" * 80)
    
    token = await get_auth_token()
    if not token:
        print("Failed to get auth token")
        return
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    params = {}
    if tenant_id:
        params["tenant_id"] = tenant_id
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_BASE_URL}/api/videos/storage/raw-videos",
            headers=headers,
            params=params
        )
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nResponse Summary:")
            print(f"  - Total items: {len(data.get('items', []))}")
            print(f"  - Tenants available: {len(data.get('tenants', []))}")
            
            print(f"\nTenants:")
            for tenant in data.get('tenants', []):
                print(f"  - {tenant.get('name')} (ID: {tenant.get('id')})")
            
            print(f"\nItems:")
            for item in data.get('items', []):
                print(f"  - Key: {item.get('key')}")
                print(f"    Filename: {item.get('filename')}")
                print(f"    Original Filename: {item.get('original_filename')}")
                print(f"    Video ID: {item.get('video_id')}")
                print(f"    Tenant ID: {item.get('tenant_id')}")
                print(f"    Tenant Name: {item.get('tenant_name')}")
                print(f"    Size: {item.get('size')}")
                print()
            
            # Save full response to file
            output_file = Path(__file__).parent / "test_raw_videos_api_response.json"
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"\nFull response saved to: {output_file}")
        else:
            print(f"\nError: {response.text}")


async def main():
    """Main test function."""
    print("Raw Videos API Test Script")
    print("=" * 80)
    
    # Test S3 connection
    if not test_s3_connection():
        print("\nS3 connection test failed. Continuing with API tests...")
    
    # Test API without filter
    await test_api_endpoint()
    
    # Test API with Farm1 filter (if we can get the tenant ID)
    # This would require querying the database or API first
    print("\n" + "=" * 80)
    print("Test completed")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

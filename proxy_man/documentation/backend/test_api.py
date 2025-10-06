#!/usr/bin/env python3
"""
Test script to verify the API is working correctly with Typesense.
Run Typesense first, then run this script.
"""

import requests
import json
import time

API_URL = "http://localhost:8000"

def test_health():
    """Test API health endpoint"""
    print("Testing API health...")
    try:
        response = requests.get(f"{API_URL}/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API Status: {data.get('status', 'unknown')}")
            if 'index_exists' in data:
                print(f"✓ Index Exists: {data['index_exists']}")
            if 'schema_version' in data:
                print(f"✓ Schema Version: {data['schema_version']}")
                print(f"✓ Schema Compatible: {data.get('schema_compatible', False)}")
            if 'approximate_documents' in data:
                print(f"✓ Approximate Documents: {data['approximate_documents']}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Is it running?")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_pages():
    """Test pages endpoint"""
    print("\nTesting pages endpoint...")
    try:
        response = requests.get(f"{API_URL}/api/pages")
        if response.status_code == 200:
            pages = response.json()
            print(f"✓ Found {len(pages)} pages")
            if pages:
                print(f"  Sample page: {pages[0].get('title', 'Unknown')}")
            return True
        else:
            print(f"✗ Pages endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_search():
    """Test search endpoint"""
    print("\nTesting search endpoint...")
    test_queries = ["documentation", "api", "test"]

    for query in test_queries:
        try:
            response = requests.get(f"{API_URL}/api/search", params={"q": query})
            if response.status_code == 200:
                results = response.json()
                print(f"✓ Search for '{query}': {len(results)} results")
                if results:
                    print(f"  Top result: {results[0].get('title', 'Unknown')}")
            else:
                print(f"✗ Search failed for '{query}': {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Error searching for '{query}': {e}")
            return False
    return True

def test_search_categories():
    """Test search with category filters"""
    print("\nTesting category filters...")
    categories = ["heading", "content", "code"]

    for category in categories:
        try:
            response = requests.get(f"{API_URL}/api/search",
                                   params={"q": "test", "category": category})
            if response.status_code == 200:
                results = response.json()
                print(f"✓ Search category '{category}': {len(results)} results")
            else:
                print(f"✗ Category search failed for '{category}': {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Error with category '{category}': {e}")
            return False
    return True

def main():
    print("=" * 50)
    print("Documentation API Test Suite")
    print("=" * 50)

    # Check if API is running
    if not test_health():
        print("\n⚠️  API is not running.")
        print("To start:")
        print("1. Install dependencies: uv pip install -e .")
        print("2. Start API: python main.py")
        return

    # Test other endpoints
    test_pages()
    test_search()
    test_search_categories()

    print("\n" + "=" * 50)
    print("Test suite completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()
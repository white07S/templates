#!/usr/bin/env python3
"""
Quick Test Script to Demonstrate the Search Engine
"""

import os
import time
import subprocess
import sys

def run_command(cmd):
    """Run a shell command and return output"""
    print(f"\n📍 Running: {cmd}")
    print("-" * 60)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print(f"Error: {result.stderr}")
    return result.returncode

def main():
    print("=" * 60)
    print("SEARCH ENGINE SYSTEM TEST")
    print("=" * 60)

    # Step 1: Check if data exists
    data_file = "data/mock_data_1m.parquet"
    indices_dir = "indices"

    if not os.path.exists(data_file):
        print("\n1️⃣ Generating test data (1M rows)...")
        if run_command("python mock_data_generator.py") != 0:
            print("Failed to generate test data!")
            return 1
    else:
        print(f"\n✓ Test data already exists at {data_file}")

    # Step 2: Build indices if not exists
    if not os.path.exists(indices_dir) or not os.listdir(indices_dir):
        print("\n2️⃣ Building indices...")
        if run_command(f"python engine.py --data-path {data_file} --batch-size 50000") != 0:
            print("Failed to build indices!")
            return 1
    else:
        print(f"\n✓ Indices already exist at {indices_dir}")

    # Step 3: Test different search modes
    print("\n3️⃣ Testing search modes...")

    # Test dataframe mode
    print("\n📌 Test 1: Dataframe Search")
    run_command('python search.py --mode dataframe --column-value-pairs category_1:category_10 --output-format summary')

    # Test keyword mode
    print("\n📌 Test 2: Keyword Search")
    run_command('python search.py --mode keyword --keywords data optimization algorithm --output-format summary')

    # Test vector mode
    print("\n📌 Test 3: Vector Search")
    run_command('python search.py --mode vector --text-for-vector "search engine performance optimization" --output-format summary')

    # Test hybrid mode
    print("\n📌 Test 4: Hybrid Search")
    run_command('python search.py --mode hybrid --keywords database --text-for-vector "indexing algorithms" --output-format summary')

    # Step 4: Run mini benchmark
    print("\n4️⃣ Running performance benchmark (10 queries per mode)...")
    run_command("python benchmark.py --num-queries 10")

    print("\n" + "=" * 60)
    print("✅ SYSTEM TEST COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("- For full 10M row test: python engine.py --generate-data --num-rows 10000000")
    print("- For full benchmark: python benchmark.py --num-queries 100")
    print("- For interactive search: python search.py --help")

if __name__ == "__main__":
    main()
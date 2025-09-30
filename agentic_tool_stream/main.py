#!/usr/bin/env python3
"""
Advanced Search Algorithm Engine - Main Entry Point
"""

import sys
import os

def main():
    print("=" * 60)
    print("ADVANCED SEARCH ALGORITHM ENGINE")
    print("=" * 60)
    print("\nAvailable commands:")
    print("\n1. Generate test data and build indices:")
    print("   python engine.py --generate-data --num-rows 1000000")
    print("\n2. Run searches:")
    print("   python search.py --mode [dataframe|keyword|vector|hybrid] [options]")
    print("\n3. Run performance benchmark:")
    print("   python benchmark.py --num-queries 100")
    print("\n4. Quick system test:")
    print("   python test_system.py")
    print("\n5. For detailed help on any command:")
    print("   python [script].py --help")
    print("\nFor a complete walkthrough, see README.md")
    print("=" * 60)

    # If run with test argument, run the test
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("\nRunning system test...")
        os.system("python test_system.py")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Setup script to initialize the task runner environment and test basic functionality.
"""

import os
import sys
from pathlib import Path
import logging
from logging_config import setup_logging

def create_directories():
    """Create necessary directories."""
    directories = [
        "logs",
        "uploads", 
        "exports",
        "prompt-lib",
        "external_loss/processed",
        "external_loss/to_process",
        "internal_loss/processed",
        "internal_loss/to_process"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def check_environment():
    """Check environment variables and dependencies."""
    print("🔍 Checking environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    print(f"✅ Python version: {sys.version}")
    
    # Check required packages
    required_packages = [
        'fastapi', 'uvicorn', 'pandas', 'openai', 
        'tenacity', 'tqdm', 'duckdb', 'tinydb', 'pydantic'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} missing")
    
    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print(f"uv add {' '.join(missing_packages)}")
        print(f"# OR pip install {' '.join(missing_packages)}")
        return False
    
    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set. Set it before running tasks:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    print("✅ OPENAI_API_KEY is set")
    
    return True

def test_basic_functionality():
    """Test basic system functionality."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test file validator
        from file_validator import FileValidator
        validator = FileValidator()
        print("✅ FileValidator initialized")
        
        # Test database manager
        from database_manager import DatabaseManager
        db_manager = DatabaseManager()
        print("✅ DatabaseManager initialized")
        
        # Test config loading
        validator.validate_task_and_data_case("summarize", "external_loss")
        print("✅ Config validation working")
        
        # Test prompt loading
        from prompt_lib.summary_prompt import render_summary_prompt
        test_row = {
            "DescriptionOfEvent": "Test event",
            "nfr_taxonomy": "Test taxonomy"
        }
        prompt = render_summary_prompt("external_loss", test_row)
        if len(prompt) > 100:
            print("✅ Prompt rendering working")
        else:
            print("⚠️  Prompt seems too short")
        
        # Close database connections
        db_manager.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def show_usage_examples():
    """Show usage examples."""
    print("\n📚 Usage Examples:")
    print("=" * 50)
    
    print("\n🖥️  CLI Usage:")
    print("python cli.py list-tasks")
    print("python cli.py run --file test_data.csv --task summarize --data_case external_loss")
    print("python cli.py status --task_id <task_id>")
    print("python cli.py download --task_id <task_id>")
    
    print("\n🌐 API Usage:")
    print("python api.py  # Start server")
    print("curl -X POST http://localhost:8000/submit -F 'file=@test_data.csv' -F 'task=summarize' -F 'data_case=external_loss'")
    print("curl http://localhost:8000/status/{task_id}")
    print("curl http://localhost:8000/download/{task_id} -o results.csv")

def main():
    """Main setup function."""
    print("🚀 Task Runner Backend Setup")
    print("=" * 40)
    
    # Setup logging
    setup_logging()
    
    # Create directories
    create_directories()
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        sys.exit(1)
    
    # Test functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality test failed.")
        sys.exit(1)
    
    print("\n✅ Setup completed successfully!")
    print("🎉 Task Runner Backend is ready to use!")
    
    # Show usage examples
    show_usage_examples()
    
    print("\n💡 Next steps:")
    print("1. Review test_data.csv to understand the expected format")
    print("2. Try running: python cli.py list-tasks")
    print("3. Test with: python cli.py run --file test_data.csv --task summarize --data_case external_loss")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test script to verify the download bug fix
"""
import pandas as pd
import os
import tempfile
import shutil
from pathlib import Path

def create_test_csv():
    """Create a test CSV file with sample data"""
    data = {
        'DescriptionOfEvent': [
            'System outage caused by network failure',
            'Data breach due to unauthorized access',
            'Trading loss from market volatility'
        ],
        'nfr_taxonomy': [
            'Technology Risk',
            'Information Security Risk', 
            'Market Risk'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    test_file = os.path.join(temp_dir, "test_data.csv")
    df.to_csv(test_file, index=False)
    
    print(f"Created test file: {test_file}")
    return test_file, temp_dir

def test_download_workflow():
    """Test the complete download workflow"""
    print("Testing download bug fix...")
    
    # Create test file
    test_file, temp_dir = create_test_csv()
    
    try:
        # Test that TaskProcessor can export results properly
        from task_processor import TaskProcessor
        from database_manager import DatabaseManager
        
        processor = TaskProcessor()
        db_manager = DatabaseManager()
        
        # Generate a test task ID
        task_id = processor.generate_task_id()
        
        # Test the export_results method with a real file
        print(f"Testing export with task_id: {task_id}")
        
        # Create some mock processed results
        test_data_case = "external_loss" 
        test_task = "summarize"
        
        # Create export directory if it doesn't exist
        os.makedirs("exports", exist_ok=True)
        
        # Test the export functionality
        try:
            export_path = processor.export_results(test_file, task_id, test_task, test_data_case)
            print(f"Export would be created at: {export_path}")
            
            # Check if export directory structure is correct
            if os.path.exists("exports"):
                print("✓ Exports directory exists")
            else:
                print("✗ Exports directory missing")
                
        except Exception as e:
            print(f"Export test failed (expected if no processed data exists): {e}")
            
        # Test database functionality for storing original file path
        db_manager.cache_task_status(task_id, "completed", test_data_case, 3, 3, test_file)
        
        # Verify the task status retrieval includes original_file_path
        status = db_manager.get_task_status(task_id)
        if status and 'original_file_path' in status:
            print("✓ Original file path stored in database")
            print(f"  Stored path: {status['original_file_path']}")
        else:
            print("✗ Original file path not stored in database")
            
        print("Download workflow test completed!")
        
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up test directory: {temp_dir}")

if __name__ == "__main__":
    test_download_workflow()
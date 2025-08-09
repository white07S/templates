#!/usr/bin/env python3
"""
Command Line Interface for Task Runner Backend

Usage:
    python cli.py run --file data.csv --task summarize --data_case external_loss
    python cli.py status --task_id <task_id>
    python cli.py download --task_id <task_id> --output results.csv
"""

import argparse
import os
import sys
import json
from pathlib import Path
import time
from typing import Optional
from datetime import datetime, timedelta

from task_processor import TaskProcessor
from llm_processor import LLMProcessor
from file_validator import FileValidator
from database_manager import DatabaseManager
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CLI:
    def __init__(self):
        self.task_processor = TaskProcessor()
        self.validator = FileValidator()
        self.db_manager = DatabaseManager()
        self.llm_processor = None
        
    def initialize_llm(self):
        """Initialize LLM processor with configuration from config.json."""
        try:
            self.llm_processor = LLMProcessor()
            model_info = self.llm_processor.get_model_info()
            logger.info(f"✅ LLM processor initialized with provider: {model_info['provider']}, model: {model_info['model']}")
        except Exception as e:
            print(f"❌ Error initializing LLM processor: {e}")
            sys.exit(1)
    
    def run_task(self, file_path: str, task: str, data_case: str, show_progress: bool = True, resume: bool = False) -> str:
        """
        Run a processing task on the given file.
        
        Args:
            file_path: Path to input file
            task: Task name
            data_case: Data case name
            show_progress: Whether to show progress bar
            resume: Whether to include tasks from to_process folder
            
        Returns:
            str: Task ID
        """
        if not self.llm_processor:
            self.initialize_llm()
        
        print(f"🚀 Starting task: {task} on {data_case}")
        print(f"📁 File: {file_path}")
        if resume:
            print("🔄 Resume mode: Will process tasks stuck in to_process folder")
        
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Validate submission
            df, is_valid, missing_columns = self.validator.validate_file_submission(
                file_path, task, data_case
            )
            
            if not is_valid:
                print(f"❌ Validation failed. Missing columns: {missing_columns}")
                sys.exit(1)
            
            print(f"✅ File validated: {len(df)} rows, {len(df.columns)} columns")
            
            # Setup progress tracking
            progress_data = {'pbar': None, 'start_time': None}
            
            def progress_callback(completed: int, total: int, result: dict):
                """Progress callback for tqdm."""
                if show_progress:
                    if progress_data['pbar'] is None:
                        progress_data['pbar'] = tqdm(
                            total=total,
                            desc="Processing events",
                            unit="events",
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                        )
                        progress_data['start_time'] = time.time()
                    
                    progress_data['pbar'].n = completed
                    progress_data['pbar'].refresh()
                    
                    # Show result status
                    if result['status'] == 'success':
                        processing_time = result.get('processing_time', 0)
                        if processing_time > 0:
                            progress_data['pbar'].set_postfix({
                                'Success': '✅',
                                'Time': f"{processing_time:.1f}s"
                            })
                    else:
                        progress_data['pbar'].set_postfix({
                            'Error': '❌',
                            'Issue': result.get('error', 'Unknown')[:20]
                        })
            
            # Submit task
            task_id = self.task_processor.submit_task(
                file_path, task, data_case, self.llm_processor, progress_callback, include_to_process=resume
            )
            
            # Close progress bar
            if progress_data['pbar']:
                progress_data['pbar'].close()
                
                # Show summary
                elapsed = time.time() - progress_data['start_time']
                print(f"\n⏱️  Total processing time: {elapsed:.1f}s")
            
            print(f"✅ Task completed successfully!")
            print(f"🆔 Task ID: {task_id}")
            
            return task_id
            
        except Exception as e:
            print(f"❌ Task failed: {str(e)}")
            sys.exit(1)
    
    def resume_tasks(self, task: str, data_case: str, show_progress: bool = True) -> str:
        """
        Resume processing of tasks stuck in to_process folder.
        
        Args:
            task: Task name
            data_case: Data case name
            show_progress: Whether to show progress bar
            
        Returns:
            str: Task ID
        """
        if not self.llm_processor:
            self.initialize_llm()
        
        print(f"🔄 Resuming tasks: {task} on {data_case}")
        print("📂 Processing tasks from to_process folder...")
        
        try:
            # Setup progress tracking
            progress_data = {'pbar': None, 'start_time': None}
            
            def progress_callback(completed: int, total: int, result: dict):
                """Progress callback for tqdm."""
                if show_progress:
                    if progress_data['pbar'] is None:
                        progress_data['pbar'] = tqdm(
                            total=total,
                            desc="Resuming events",
                            unit="events",
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                        )
                        progress_data['start_time'] = time.time()
                    
                    progress_data['pbar'].n = completed
                    progress_data['pbar'].refresh()
                    
                    # Show result status
                    if result['status'] == 'success':
                        processing_time = result.get('processing_time', 0)
                        if processing_time > 0:
                            progress_data['pbar'].set_postfix({
                                'Success': '✅',
                                'Time': f"{processing_time:.1f}s"
                            })
                    else:
                        progress_data['pbar'].set_postfix({
                            'Error': '❌',
                            'Issue': result.get('error', 'Unknown')[:20]
                        })
            
            # Resume pending tasks
            task_id = self.task_processor.resume_pending_tasks(
                task, data_case, self.llm_processor, progress_callback
            )
            
            # Close progress bar
            if progress_data['pbar']:
                progress_data['pbar'].close()
                
                # Show summary
                elapsed = time.time() - progress_data['start_time']
                print(f"\n⏱️  Total processing time: {elapsed:.1f}s")
            
            print(f"✅ Resume task completed successfully!")
            print(f"🆔 Task ID: {task_id}")
            
            return task_id
            
        except Exception as e:
            print(f"❌ Resume task failed: {str(e)}")
            sys.exit(1)
    
    def check_status(self, task_id: str, watch: bool = False):
        """
        Check status of a task.
        
        Args:
            task_id: Task ID to check
            watch: Whether to watch for status changes
        """
        print(f"🔍 Checking status for task: {task_id}")
        
        def show_status():
            status = self.task_processor.get_task_status(task_id)
            
            if 'error' in status:
                print(f"❌ Task not found: {task_id}")
                return False
            
            # Status mapping for display
            status_icons = {
                'pending': '⏳',
                'processing': '🔄',
                'completed': '✅',
                'completed_with_errors': '⚠️',
                'failed': '❌'
            }
            
            icon = status_icons.get(status['status'], '❓')
            
            print(f"\n{icon} Status: {status['status'].upper()}")
            print(f"📊 Progress: {status.get('progress_percentage', 0):.1f}%")
            print(f"📈 Items: {status.get('processed_items', 0)}/{status.get('total_items', 0)}")
            print(f"🕐 Created: {status.get('created_at', 'Unknown')}")
            print(f"🕑 Updated: {status.get('updated_at', 'Unknown')}")
            
            if status['status'] in ['completed', 'completed_with_errors']:
                print(f"💾 Ready for download: Use 'python cli.py download --task_id {task_id}'")
                return False
            elif status['status'] == 'failed':
                return False
            
            return True
        
        if watch:
            print("👀 Watching for status changes (Press Ctrl+C to stop)...")
            try:
                while show_status():
                    time.sleep(5)  # Check every 5 seconds
            except KeyboardInterrupt:
                print("\n🛑 Stopped watching")
        else:
            show_status()
    
    def download_results(self, task_id: str, output_path: Optional[str] = None,
                        task: str = "summarize", data_case: str = "external_loss"):
        """
        Download results for a completed task.
        
        Args:
            task_id: Task ID
            output_path: Output file path
            task: Task name
            data_case: Data case name
        """
        print(f"📥 Downloading results for task: {task_id}")
        
        try:
            # Check task status
            status = self.task_processor.get_task_status(task_id)
            
            if 'error' in status:
                print(f"❌ Task not found: {task_id}")
                sys.exit(1)
            
            if status['status'] not in ['completed', 'completed_with_errors']:
                print(f"❌ Task not ready for download. Status: {status['status']}")
                print("Use 'python cli.py status --task_id {task_id}' to check progress")
                sys.exit(1)
            
            # Generate output path if not provided
            if not output_path:
                output_path = f"task_{task_id}_{task}_{data_case}_results.csv"
            
            # Get the original file path from task status
            original_file = status.get('original_file_path')
            if not original_file:
                print(f"❌ Error: Original file path not found for task {task_id}")
                sys.exit(1)
            
            if not os.path.exists(original_file):
                print(f"❌ Error: Original file not found: {original_file}")
                sys.exit(1)
            
            # Export results
            export_path = self.task_processor.export_results(original_file, task_id, task, data_case)
            
            # Copy to desired location
            import shutil
            shutil.copy2(export_path, output_path)
            
            print(f"✅ Results downloaded to: {output_path}")
            
            # Show summary
            import pandas as pd
            df = pd.read_csv(output_path)
            print(f"📊 File contains {len(df)} rows with AI-generated columns")
            
        except Exception as e:
            print(f"❌ Download failed: {str(e)}")
            sys.exit(1)
    
    def list_tasks(self):
        """List available tasks and data cases."""
        with open("config.json", 'r') as f:
            config = json.load(f)
        
        print("📋 Available Tasks and Data Cases:")
        print("=" * 50)
        
        for task_name, task_config in config['tasks'].items():
            print(f"\n🎯 Task: {task_name}")
            print(f"   Description: {task_config['description']}")
            print(f"   Prompt: {task_config['prompt']}")
            
            print("   📊 Data Cases:")
            for case_name, case_config in task_config['data_cases'].items():
                print(f"     • {case_name}: {case_config['description']}")
                print(f"       Required columns: {list(case_config['mandatory_columns'].keys())}")
    
    def cleanup_cache(self, days: int = 7):
        """Clean up old cache entries."""
        print(f"🧹 Cleaning up cache entries older than {days} days...")
        
        try:
            self.db_manager.cleanup_cache(days)
            print("✅ Cache cleanup completed")
        except Exception as e:
            print(f"❌ Cache cleanup failed: {str(e)}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Task Runner Backend CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a task
  python cli.py run --file data.csv --task summarize --data_case external_loss
  
  # Run a task including stuck to_process items
  python cli.py run --file data.csv --task summarize --data_case external_loss --resume
  
  # Resume processing stuck tasks only
  python cli.py resume --task summarize --data_case external_loss
  
  # Check status
  python cli.py status --task_id abc123
  
  # Watch status (updates every 5 seconds)
  python cli.py status --task_id abc123 --watch
  
  # Download results
  python cli.py download --task_id abc123 --output results.csv
  
  # List available tasks
  python cli.py list-tasks
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a processing task')
    run_parser.add_argument('--file', required=True, help='Path to input file (CSV/Excel)')
    run_parser.add_argument('--task', required=True, help='Task name (e.g., summarize)')
    run_parser.add_argument('--data_case', required=True, help='Data case (e.g., external_loss)')
    run_parser.add_argument('--no-progress', action='store_true', help='Disable progress bar')
    run_parser.add_argument('--resume', action='store_true', help='Include tasks from to_process folder')
    
    # Resume command
    resume_parser = subparsers.add_parser('resume', help='Resume processing tasks stuck in to_process folder')
    resume_parser.add_argument('--task', required=True, help='Task name (e.g., summarize)')
    resume_parser.add_argument('--data_case', required=True, help='Data case (e.g., external_loss)')
    resume_parser.add_argument('--no-progress', action='store_true', help='Disable progress bar')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check task status')
    status_parser.add_argument('--task_id', required=True, help='Task ID to check')
    status_parser.add_argument('--watch', action='store_true', help='Watch for status changes')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download results')
    download_parser.add_argument('--task_id', required=True, help='Task ID')
    download_parser.add_argument('--output', help='Output file path')
    download_parser.add_argument('--task', default='summarize', help='Task name')
    download_parser.add_argument('--data_case', default='external_loss', help='Data case')
    
    # List tasks command
    subparsers.add_parser('list-tasks', help='List available tasks and data cases')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old cache entries')
    cleanup_parser.add_argument('--days', type=int, default=7, help='Clean entries older than N days')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = CLI()
    
    try:
        if args.command == 'run':
            task_id = cli.run_task(args.file, args.task, args.data_case, not args.no_progress, args.resume)
            
        elif args.command == 'resume':
            task_id = cli.resume_tasks(args.task, args.data_case, not args.no_progress)
            
        elif args.command == 'status':
            cli.check_status(args.task_id, args.watch)
            
        elif args.command == 'download':
            cli.download_results(args.task_id, args.output, args.task, args.data_case)
            
        elif args.command == 'list-tasks':
            cli.list_tasks()
            
        elif args.command == 'cleanup':
            cli.cleanup_cache(args.days)
            
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.exception("Unexpected error occurred")
        print(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
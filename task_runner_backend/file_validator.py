import pandas as pd
import json
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class FileValidator:
    def __init__(self, config_path: str = "config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def validate_file_format(self, file_path: str) -> pd.DataFrame:
        """
        Validate and load the uploaded file (CSV/Excel).
        
        Args:
            file_path: Path to the uploaded file
            
        Returns:
            pd.DataFrame: Loaded dataframe
            
        Raises:
            ValueError: If file format is unsupported or corrupted
        """
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format. Only CSV and Excel files are supported.")
            
            if df.empty:
                raise ValueError("File is empty or could not be parsed.")
                
            logger.info(f"Successfully loaded file with {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {str(e)}")
            raise ValueError(f"Failed to load file: {str(e)}")
    
    def validate_task_and_data_case(self, task: str, data_case: str) -> bool:
        """
        Validate if the task and data_case exist in config.
        
        Args:
            task: Task name
            data_case: Data case name
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If task or data_case is invalid
        """
        if task not in self.config['tasks']:
            raise ValueError(f"Task '{task}' not found. Available tasks: {list(self.config['tasks'].keys())}")
        
        if data_case not in self.config['tasks'][task]['data_cases']:
            available_cases = list(self.config['tasks'][task]['data_cases'].keys())
            raise ValueError(f"Data case '{data_case}' not found for task '{task}'. Available cases: {available_cases}")
        
        return True
    
    def validate_mandatory_columns(self, df: pd.DataFrame, task: str, data_case: str) -> Tuple[bool, List[str]]:
        """
        Check if all mandatory columns are present in the dataframe.
        
        Args:
            df: Input dataframe
            task: Task name
            data_case: Data case name
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, missing_columns)
        """
        mandatory_columns = self.config['tasks'][task]['data_cases'][data_case]['mandatory_columns']
        missing_columns = []
        
        for column in mandatory_columns.keys():
            if column not in df.columns:
                missing_columns.append(column)
        
        is_valid = len(missing_columns) == 0
        
        if not is_valid:
            logger.warning(f"Missing mandatory columns: {missing_columns}")
        else:
            logger.info(f"All mandatory columns present for {task}/{data_case}")
            
        return is_valid, missing_columns
    
    def get_mandatory_columns_info(self, task: str, data_case: str) -> Dict[str, str]:
        """
        Get information about mandatory columns for a task/data_case.
        
        Args:
            task: Task name
            data_case: Data case name
            
        Returns:
            Dict[str, str]: Column name -> description mapping
        """
        return self.config['tasks'][task]['data_cases'][data_case]['mandatory_columns']
    
    def validate_file_submission(self, file_path: str, task: str, data_case: str) -> Tuple[pd.DataFrame, bool, List[str]]:
        """
        Complete validation pipeline for file submission.
        
        Args:
            file_path: Path to the uploaded file
            task: Task name
            data_case: Data case name
            
        Returns:
            Tuple[pd.DataFrame, bool, List[str]]: (dataframe, is_valid, missing_columns)
        """
        # Validate task and data case
        self.validate_task_and_data_case(task, data_case)
        
        # Load and validate file
        df = self.validate_file_format(file_path)
        
        # Check mandatory columns
        is_valid, missing_columns = self.validate_mandatory_columns(df, task, data_case)
        
        return df, is_valid, missing_columns
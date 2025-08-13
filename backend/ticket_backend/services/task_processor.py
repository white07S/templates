import os
import json
import pandas as pd
from typing import Dict, List
from ticket_backend.utils.file_handler import read_file_data, save_result_file

class TaskProcessor:
    def __init__(self):
        pass
    
    def process_task(self, task_dir: str) -> bool:
        try:
            task_info_path = os.path.join(task_dir, "task_info.json")
            if not os.path.exists(task_info_path):
                return False
            
            with open(task_info_path, 'r') as f:
                task_info = json.load(f)
            
            file_path = task_info["file_path"]
            data_type = task_info["data_type"]
            tasks = task_info["tasks"]
            
            df = read_file_data(file_path)
            
            processed_df = self._execute_tasks(df, data_type, tasks)
            
            completed_dir = os.path.join(task_dir, "completed")
            os.makedirs(completed_dir, exist_ok=True)
            
            result_filename = f"result_{os.path.basename(file_path)}"
            result_path = os.path.join(completed_dir, result_filename)
            
            save_result_file(processed_df, result_path)
            
            return True
            
        except Exception as e:
            print(f"Error processing task in {task_dir}: {str(e)}")
            return False
    
    def _execute_tasks(self, df: pd.DataFrame, data_type: str, tasks: List[str]) -> pd.DataFrame:
        processed_df = df.copy()
        
        for task in tasks:
            processed_df = self._execute_single_task(processed_df, data_type, task)
        
        return processed_df
    
    def _execute_single_task(self, df: pd.DataFrame, data_type: str, task: str) -> pd.DataFrame:
        if task == "5Ws extraction":
            return self._extract_5ws(df, data_type)
        elif task == "similarity":
            return self._calculate_similarity(df, data_type)
        elif task == "taxonomy_mapping":
            return self._map_taxonomy(df, data_type)
        elif task == "root_cause_mapping":
            return self._map_root_cause(df, data_type)
        elif task == "ai_insights":
            return self._generate_ai_insights(df, data_type)
        else:
            return df
    
    def _extract_5ws(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        df["5Ws_Who"] = "Sample Who"
        df["5Ws_What"] = "Sample What"
        df["5Ws_When"] = "Sample When"
        df["5Ws_Where"] = "Sample Where"
        df["5Ws_Why"] = "Sample Why"
        return df
    
    def _calculate_similarity(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        df["Similarity_Score"] = 0.85
        df["Similar_Items"] = "Item1, Item2, Item3"
        return df
    
    def _map_taxonomy(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        df["Taxonomy_Category"] = "Category A"
        df["Taxonomy_Subcategory"] = "Subcategory 1"
        return df
    
    def _map_root_cause(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        df["Root_Cause"] = "Process Failure"
        df["Root_Cause_Category"] = "Operational"
        return df
    
    def _generate_ai_insights(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        df["AI_Risk_Score"] = 7.5
        df["AI_Recommendation"] = "Implement additional controls"
        df["AI_Impact_Assessment"] = "Medium Risk"
        return df

task_processor = TaskProcessor()
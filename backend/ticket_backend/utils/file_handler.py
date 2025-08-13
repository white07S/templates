import os
import pandas as pd
from fastapi import UploadFile
from typing import List

async def save_uploaded_file(file: UploadFile, directory: str) -> str:
    file_path = os.path.join(directory, file.filename)
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    return file_path

def validate_file_columns(file_path: str, mandatory_columns: List[str]) -> bool:
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return False
        
        file_columns = df.columns.tolist()
        
        for col in mandatory_columns:
            if col not in file_columns:
                return False
        
        return True
        
    except Exception:
        return False

def read_file_data(file_path: str) -> pd.DataFrame:
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")

def save_result_file(data: pd.DataFrame, output_path: str, format: str = "xlsx"):
    if format == "csv":
        data.to_csv(output_path, index=False)
    else:
        data.to_excel(output_path, index=False)
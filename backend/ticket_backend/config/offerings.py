from typing import Dict, List

SUPPORTED_DATA_TYPES = {
    "controls": {
        "mandatory_columns": ["Control Description", "Control ID"],
        "tasks": ["5Ws extraction", "similarity", "taxonomy_mapping", "root_cause_mapping", "ai_insights"],
        "supported_formats": ["xlsx", "csv"]
    },
    "issues": {
        "mandatory_columns": ["Issue Description", "Issue ID"],
        "tasks": ["taxonomy_mapping", "root_cause_mapping", "ai_insights"],
        "supported_formats": ["xlsx", "csv"]
    },
    "external_loss": {
        "mandatory_columns": ["Loss Description", "Loss ID"],
        "tasks": ["taxonomy_mapping", "root_cause_mapping", "ai_insights"],
        "supported_formats": ["xlsx", "csv"]
    },
    "internal_loss": {
        "mandatory_columns": ["Loss Description", "Loss ID"],
        "tasks": ["taxonomy_mapping", "root_cause_mapping", "ai_insights"],
        "supported_formats": ["xlsx", "csv"]
    },
    "orx_scenarios": {
        "mandatory_columns": ["Scenario Description", "Scenario ID"],
        "tasks": ["taxonomy_mapping", "root_cause_mapping", "ai_insights"],
        "supported_formats": ["xlsx", "csv"]
    }
}

def get_available_data_types() -> List[Dict[str, any]]:
    return [
        {
            "name": data_type,
            "tasks": config["tasks"]
        }
        for data_type, config in SUPPORTED_DATA_TYPES.items()
    ]

def get_mandatory_columns(data_type: str) -> List[str]:
    return SUPPORTED_DATA_TYPES.get(data_type, {}).get("mandatory_columns", [])

def get_supported_tasks(data_type: str) -> List[str]:
    return SUPPORTED_DATA_TYPES.get(data_type, {}).get("tasks", [])

def is_valid_data_type(data_type: str) -> bool:
    return data_type in SUPPORTED_DATA_TYPES

def is_valid_task(data_type: str, task: str) -> bool:
    return task in get_supported_tasks(data_type)

def get_supported_formats(data_type: str) -> List[str]:
    return SUPPORTED_DATA_TYPES.get(data_type, {}).get("supported_formats", [])
from tinydb import TinyDB, Query
from pathlib import Path
import json
from datetime import datetime

# Create data directory if it doesn't exist
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Initialize databases
db = TinyDB(data_dir / "glossary.json")
temp_db = TinyDB(data_dir / "temp_glossary.json")

def get_mock_data():
    """Returns mock glossary data"""
    return {
        "terms": [
            {
                "id": 1,
                "term": "API",
                "definition": "Application Programming Interface",
                "synonyms": ["Application Interface", "Programming Interface"],
                "createdAt": datetime.now().isoformat(),
                "updatedAt": datetime.now().isoformat()
            },
            {
                "id": 2,
                "term": "JSON",
                "definition": "JavaScript Object Notation",
                "synonyms": ["JavaScript Notation"],
                "createdAt": datetime.now().isoformat(),
                "updatedAt": datetime.now().isoformat()
            },
            {
                "id": 3,
                "term": "REST",
                "definition": "Representational State Transfer",
                "synonyms": ["RESTful", "REST API"],
                "createdAt": datetime.now().isoformat(),
                "updatedAt": datetime.now().isoformat()
            },
            {
                "id": 4,
                "term": "HTTP",
                "definition": "HyperText Transfer Protocol",
                "synonyms": ["Web Protocol"],
                "createdAt": datetime.now().isoformat(),
                "updatedAt": datetime.now().isoformat()
            },
            {
                "id": 5,
                "term": "SQL",
                "definition": "Structured Query Language",
                "synonyms": ["Database Query Language", "Sequel"],
                "createdAt": datetime.now().isoformat(),
                "updatedAt": datetime.now().isoformat()
            }
        ]
    }

def initialize_database():
    """Initialize database with mock data if empty"""
    if len(db) == 0:
        mock_data = get_mock_data()
        for term in mock_data["terms"]:
            db.insert(term)
    else:
        # Migrate existing data to include synonyms field if not present
        migrate_synonyms_field()

def migrate_synonyms_field():
    """Add synonyms field to existing records that don't have it"""
    Term_query = Query()
    all_terms = db.all()
    for term in all_terms:
        if "synonyms" not in term:
            db.update({"synonyms": []}, Term_query.id == term["id"])

def get_next_id():
    """Get the next available ID for a new term"""
    all_terms = db.all()
    if not all_terms:
        return 1
    return max(term["id"] for term in all_terms) + 1
"""
Indexing module for Tantivy
Handles indexing of nested JSON documents
"""

import os
import orjson
import tantivy
from typing import Dict, Any, List
import config


class JSONIndexer:
    def __init__(self, index_dir: str = config.INDEX_DIR):
        """Initialize the Tantivy indexer"""
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)

        # Create schema
        self.schema_builder = tantivy.SchemaBuilder()

        # Add fields
        # Use "en_stem" tokenizer for filename to enable partial matching and case-insensitive search
        self.schema_builder.add_text_field(config.FIELD_FILENAME, stored=True, tokenizer_name="en_stem")
        self.schema_builder.add_text_field(config.FIELD_CONTENT, stored=False, tokenizer_name="en_stem")
        self.schema_builder.add_text_field(config.FIELD_PATH, stored=True)
        self.schema_builder.add_json_field(config.FIELD_RAW_JSON, stored=True)

        self.schema = self.schema_builder.build()

        # Create or open index
        if not os.path.exists(os.path.join(index_dir, "meta.json")):
            self.index = tantivy.Index(self.schema, path=index_dir)
        else:
            self.index = tantivy.Index.open(index_dir)

        self.writer = None

    def flatten_json(self, obj: Any, parent_key: str = '', sep: str = '.') -> Dict[str, str]:
        """
        Recursively flatten nested JSON structure into a flat dictionary.
        Handles lists, dicts, and nested structures.
        """
        items = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}{sep}{key}" if parent_key else key
                if isinstance(value, (dict, list)):
                    items.extend(self.flatten_json(value, new_key, sep=sep).items())
                else:
                    items.append((new_key, str(value) if value is not None else ""))

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_key = f"{parent_key}[{i}]"
                if isinstance(item, (dict, list)):
                    items.extend(self.flatten_json(item, new_key, sep=sep).items())
                else:
                    items.append((new_key, str(item) if item is not None else ""))

        else:
            items.append((parent_key, str(obj) if obj is not None else ""))

        return dict(items)

    def extract_searchable_content(self, json_data: Dict) -> str:
        """
        Extract all searchable content from nested JSON.
        Returns a concatenated string of all text values.
        """
        flat_data = self.flatten_json(json_data)

        # Concatenate all values for full-text search
        content_parts = []
        for key, value in flat_data.items():
            # Add both keys and values to searchable content
            content_parts.append(key)
            if value:
                content_parts.append(value)

        return " ".join(content_parts)

    def index_document(self, filepath: str, writer = None):
        """Index a single JSON document"""
        try:
            # Read and parse JSON file
            with open(filepath, "rb") as f:
                json_data = orjson.loads(f.read())

            # Extract filename
            filename = os.path.basename(filepath)

            # Also add filename without extension for better partial matching
            filename_no_ext = os.path.splitext(filename)[0]

            # Extract searchable content
            content = self.extract_searchable_content(json_data)

            # Create document
            doc = tantivy.Document()
            # Store both full filename and name without extension for better matching
            doc.add_text(config.FIELD_FILENAME, f"{filename} {filename_no_ext}")
            doc.add_text(config.FIELD_CONTENT, content)
            doc.add_text(config.FIELD_PATH, filepath)
            doc.add_json(config.FIELD_RAW_JSON, json_data)

            # Add to index
            if writer:
                writer.add_document(doc)
            else:
                if not self.writer:
                    self.writer = self.index.writer(config.MEMORY_SIZE, config.NUM_THREADS)
                self.writer.add_document(doc)

            return True

        except Exception as e:
            print(f"Error indexing {filepath}: {e}")
            return False

    def index_directory(self, data_dir: str = config.DATA_DIR):
        """Index all JSON files in a directory"""
        json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        total_files = len(json_files)

        print(f"Found {total_files} JSON files to index")

        indexed_count = 0
        error_count = 0

        # Create a single writer for all documents
        try:
            writer = self.index.writer(config.MEMORY_SIZE, config.NUM_THREADS)
        except Exception as e:
            print(f"Error creating writer: {e}")
            return 0, total_files

        for i, filename in enumerate(json_files):
            filepath = os.path.join(data_dir, filename)

            if self.index_document(filepath, writer):
                indexed_count += 1
            else:
                error_count += 1

            # Progress update
            if (i + 1) % 100 == 0 or (i + 1) == total_files:
                print(f"Indexed {i + 1}/{total_files} files...")

        # Final commit
        writer.commit()

        print(f"Indexing complete: {indexed_count} indexed, {error_count} errors")
        self.index.reload()

        return indexed_count, error_count

    def clear_index(self):
        """Clear the existing index"""
        if self.writer:
            self.writer.commit()
            self.writer = None

        # Close the index
        self.index = None

        # Remove index directory
        if os.path.exists(self.index_dir):
            import shutil
            shutil.rmtree(self.index_dir)
            print(f"Index cleared from {self.index_dir}")

        # Recreate empty index
        os.makedirs(self.index_dir, exist_ok=True)
        self.index = tantivy.Index(self.schema, path=self.index_dir)

    def get_index_stats(self):
        """Get statistics about the index"""
        self.index.reload()
        searcher = self.index.searcher()
        num_docs = searcher.num_docs
        return {
            "num_documents": num_docs,
            "index_path": self.index_dir
        }


def main():
    """Main function for testing indexing"""
    indexer = JSONIndexer()

    # Clear existing index
    indexer.clear_index()

    # Index all JSON files
    indexed, errors = indexer.index_directory()

    # Get stats
    stats = indexer.get_index_stats()
    print(f"\nIndex Statistics:")
    print(f"  Total documents: {stats['num_documents']}")
    print(f"  Index location: {stats['index_path']}")


if __name__ == "__main__":
    main()
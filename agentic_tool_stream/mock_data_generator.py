"""
Mock Data Generator for Search Engine Testing
Generates 10M rows with 50+ columns of various data types
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from datetime import datetime, timedelta
from tqdm import tqdm
import random
import string
import os

class MockDataGenerator:
    def __init__(self, num_rows=10_000_000, num_columns=55, output_path="data/mock_data.parquet"):
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.output_path = output_path
        self.batch_size = 100_000  # Process in batches for memory efficiency

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def generate_text(self, min_words=10, max_words=600):
        """Generate random text with specified word count"""
        words = [
            "data", "search", "engine", "performance", "optimization", "query",
            "algorithm", "index", "database", "system", "architecture", "design",
            "implementation", "production", "scalability", "efficiency", "process",
            "analysis", "structure", "function", "method", "approach", "solution",
            "technology", "framework", "platform", "service", "application", "model"
        ]

        num_words = random.randint(min_words, max_words)
        return " ".join(random.choices(words, k=num_words))

    def generate_category(self, num_categories=50):
        """Generate categorical values"""
        categories = [f"category_{i}" for i in range(num_categories)]
        return random.choice(categories)

    def generate_batch(self, batch_num):
        """Generate a batch of data"""
        batch_start = batch_num * self.batch_size
        batch_end = min((batch_num + 1) * self.batch_size, self.num_rows)
        batch_rows = batch_end - batch_start

        data = {}

        # ID column
        data['id'] = np.arange(batch_start, batch_end)

        # Long text columns (5 columns with 500+ words)
        for i in range(5):
            column_name = f'long_text_{i+1}'
            data[column_name] = [self.generate_text(500, 800) for _ in range(batch_rows)]

        # Medium text columns (10 columns with 50-200 words)
        for i in range(10):
            column_name = f'medium_text_{i+1}'
            data[column_name] = [self.generate_text(50, 200) for _ in range(batch_rows)]

        # Short text columns (10 columns with 5-50 words)
        for i in range(10):
            column_name = f'short_text_{i+1}'
            data[column_name] = [self.generate_text(5, 50) for _ in range(batch_rows)]

        # Categorical columns (10 columns)
        for i in range(10):
            column_name = f'category_{i+1}'
            data[column_name] = [self.generate_category() for _ in range(batch_rows)]

        # Numerical columns (10 columns)
        for i in range(5):
            column_name = f'integer_{i+1}'
            data[column_name] = np.random.randint(0, 1000000, batch_rows)

        for i in range(5):
            column_name = f'float_{i+1}'
            data[column_name] = np.random.uniform(0, 1000, batch_rows)

        # Date columns (3 columns)
        base_date = datetime(2020, 1, 1)
        for i in range(3):
            column_name = f'date_{i+1}'
            data[column_name] = [base_date + timedelta(days=random.randint(0, 1825))
                                for _ in range(batch_rows)]

        # Boolean columns (3 columns)
        for i in range(3):
            column_name = f'boolean_{i+1}'
            data[column_name] = np.random.choice([True, False], batch_rows)

        # Columns with NULL/NaN values (3 columns)
        for i in range(3):
            column_name = f'nullable_{i+1}'
            values = [self.generate_text(10, 50) if random.random() > 0.3 else None
                     for _ in range(batch_rows)]
            data[column_name] = values

        # Email column
        data['email'] = [f"user{batch_start+i}@example.com" for i in range(batch_rows)]

        # URL column
        data['url'] = [f"https://example.com/page/{batch_start+i}" for i in range(batch_rows)]

        return pd.DataFrame(data)

    def generate(self):
        """Generate the complete dataset and save to parquet"""
        print(f"Generating {self.num_rows:,} rows with {self.num_columns} columns...")

        num_batches = (self.num_rows + self.batch_size - 1) // self.batch_size

        # Initialize parquet writer
        writer = None
        schema = None

        for batch_num in tqdm(range(num_batches), desc="Generating batches"):
            batch_df = self.generate_batch(batch_num)

            # Convert to PyArrow Table
            table = pa.Table.from_pandas(batch_df)

            if writer is None:
                # Create writer with the schema from first batch
                schema = table.schema
                writer = pq.ParquetWriter(self.output_path, schema, compression='snappy')

            writer.write_table(table)

        if writer:
            writer.close()

        print(f"Mock data generated and saved to {self.output_path}")

        # Verify the file
        parquet_file = pq.ParquetFile(self.output_path)
        print(f"Total rows in file: {parquet_file.metadata.num_rows:,}")
        print(f"Total columns: {len(parquet_file.schema)}")
        print(f"File size: {os.path.getsize(self.output_path) / (1024**3):.2f} GB")

if __name__ == "__main__":
    # Generate smaller dataset for testing (1M rows)
    generator = MockDataGenerator(num_rows=1_000_000, output_path="data/mock_data_1m.parquet")
    generator.generate()
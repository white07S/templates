"""Generate mock dataset for testing the search engine."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import random
import string
from datetime import datetime, timedelta
import os


def generate_random_text(min_words=10, max_words=100):
    """Generate random text with specified word count."""
    word_count = random.randint(min_words, max_words)
    words = []
    for _ in range(word_count):
        word_length = random.randint(3, 12)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_length))
        words.append(word)
    return ' '.join(words)


def generate_long_text(min_words=500, max_words=1000):
    """Generate long text for content columns."""
    word_count = random.randint(min_words, max_words)
    words = []

    # Add some common words to make it more realistic
    common_words = ["the", "and", "in", "of", "to", "a", "is", "that", "it", "for",
                    "with", "as", "was", "on", "by", "at", "from", "up", "about", "into"]

    for _ in range(word_count):
        if random.random() < 0.3:  # 30% chance of common word
            words.append(random.choice(common_words))
        else:
            word_length = random.randint(3, 12)
            word = ''.join(random.choices(string.ascii_lowercase, k=word_length))
            words.append(word)
    return ' '.join(words)


def generate_mock_dataset(num_rows=10_000_000, output_path="data/raw/data.parquet"):
    """Generate a mock dataset with specified number of rows."""

    print(f"Generating mock dataset with {num_rows:,} rows...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Define column specifications
    columns = {
        'id': 'int64',
        'title': 'string',
        'description': 'string',
        'category': 'category',
        'subcategory': 'category',
        'content': 'text_long',
        'abstract': 'text_long',
        'summary': 'text_long',
        'body': 'text_long',
        'metadata': 'text_long',
        'author': 'string',
        'organization': 'string',
        'department': 'category',
        'tags': 'string',
        'keywords': 'string',
        'status': 'category',
        'priority': 'category',
        'type': 'category',
        'source': 'string',
        'url': 'string',
        'email': 'string',
        'phone': 'string',
        'address': 'string',
        'city': 'string',
        'state': 'category',
        'country': 'category',
        'postal_code': 'string',
        'price': 'float64',
        'quantity': 'int64',
        'discount': 'float64',
        'rating': 'float64',
        'reviews_count': 'int64',
        'views_count': 'int64',
        'likes_count': 'int64',
        'shares_count': 'int64',
        'comments': 'text',
        'notes': 'text',
        'created_date': 'datetime',
        'modified_date': 'datetime',
        'published_date': 'datetime',
        'expired_date': 'datetime',
        'is_active': 'bool',
        'is_featured': 'bool',
        'is_archived': 'bool',
        'is_deleted': 'bool',
        'version': 'string',
        'language': 'category',
        'reference_id': 'string',
        'parent_id': 'string',
        'extra_field_1': 'string',
        'extra_field_2': 'string',
        'extra_field_3': 'string',
    }

    # Categories for categorical columns
    categories = {
        'category': ['Technology', 'Science', 'Business', 'Health', 'Education',
                    'Entertainment', 'Sports', 'Politics', 'Finance', 'Travel'],
        'subcategory': ['SubA', 'SubB', 'SubC', 'SubD', 'SubE', 'SubF'],
        'department': ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations'],
        'status': ['Active', 'Pending', 'Completed', 'Cancelled', 'Draft'],
        'priority': ['Low', 'Medium', 'High', 'Critical'],
        'type': ['TypeA', 'TypeB', 'TypeC', 'TypeD'],
        'state': ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'],
        'country': ['USA', 'Canada', 'UK', 'Germany', 'France', 'Japan', 'Australia'],
        'language': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko']
    }

    # Generate data in chunks to manage memory
    chunk_size = 100000
    num_chunks = (num_rows + chunk_size - 1) // chunk_size

    # First, create the parquet writer
    first_chunk = True
    writer = None

    for chunk_idx in tqdm(range(num_chunks), desc="Generating chunks"):
        current_chunk_size = min(chunk_size, num_rows - chunk_idx * chunk_size)

        data = {}

        for col_name, col_type in columns.items():
            if col_type == 'int64':
                data[col_name] = np.random.randint(0, 1000000, current_chunk_size)

            elif col_type == 'float64':
                data[col_name] = np.random.uniform(0, 1000, current_chunk_size)

            elif col_type == 'string':
                data[col_name] = [generate_random_text(2, 10) for _ in range(current_chunk_size)]

            elif col_type == 'text':
                data[col_name] = [generate_random_text(20, 100) for _ in range(current_chunk_size)]

            elif col_type == 'text_long':
                # These are the columns with 500+ words
                data[col_name] = [generate_long_text(500, 800) for _ in range(current_chunk_size)]

            elif col_type == 'category':
                if col_name in categories:
                    data[col_name] = np.random.choice(categories[col_name], current_chunk_size)
                else:
                    data[col_name] = np.random.choice(['A', 'B', 'C', 'D'], current_chunk_size)

            elif col_type == 'datetime':
                start_date = datetime(2020, 1, 1)
                end_date = datetime(2024, 12, 31)
                dates = []
                for _ in range(current_chunk_size):
                    random_days = random.randint(0, (end_date - start_date).days)
                    dates.append(start_date + timedelta(days=random_days))
                data[col_name] = dates

            elif col_type == 'bool':
                data[col_name] = np.random.choice([True, False], current_chunk_size)

        # Add some null values, empty strings, and NaN values randomly
        for col_name in columns.keys():
            if columns[col_name] not in ['int64', 'bool']:
                # Randomly set some values to None or empty string
                null_mask = np.random.random(current_chunk_size) < 0.05  # 5% null values
                empty_mask = np.random.random(current_chunk_size) < 0.03  # 3% empty strings

                for i in range(current_chunk_size):
                    if null_mask[i]:
                        data[col_name][i] = None
                    elif empty_mask[i] and columns[col_name] in ['string', 'text', 'text_long']:
                        data[col_name][i] = ''

        # Create DataFrame
        df = pd.DataFrame(data)

        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df)

        # Write to parquet
        if first_chunk:
            writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
            first_chunk = False

        writer.write_table(table)

    # Close the writer
    if writer:
        writer.close()

    print(f"✓ Mock dataset saved to {output_path}")
    print(f"  - Rows: {num_rows:,}")
    print(f"  - Columns: {len(columns)}")
    print(f"  - File size: {os.path.getsize(output_path) / (1024**3):.2f} GB")


if __name__ == "__main__":
    # Generate the full 10M row dataset
    generate_mock_dataset(num_rows=10_000_000)
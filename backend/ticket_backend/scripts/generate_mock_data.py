#!/usr/bin/env python3
import argparse
import pandas as pd
import random
import os
from faker import Faker

fake = Faker()

DATA_TYPE_CONFIGS = {
    "controls": {
        "mandatory_columns": ["Control Description", "Control ID"],
        "optional_columns": ["Control Owner", "Implementation Date", "Review Date", "Status"]
    },
    "issues": {
        "mandatory_columns": ["Issue Description", "Issue ID"],
        "optional_columns": ["Severity", "Status", "Reporter", "Date Reported", "Resolution Date"]
    },
    "external_loss": {
        "mandatory_columns": ["Loss Description", "Loss ID"],
        "optional_columns": ["Loss Amount", "Currency", "Event Date", "Recovery Amount", "Business Line"]
    },
    "internal_loss": {
        "mandatory_columns": ["Loss Description", "Loss ID"],
        "optional_columns": ["Loss Amount", "Currency", "Event Date", "Recovery Amount", "Business Line"]
    },
    "orx_scenarios": {
        "mandatory_columns": ["Scenario Description", "Scenario ID"],
        "optional_columns": ["Risk Category", "Probability", "Impact", "Mitigation", "Business Line"]
    }
}

def generate_controls_data(num_records: int) -> pd.DataFrame:
    data = []
    for i in range(num_records):
        record = {
            "Control ID": f"CTRL-{i+1:04d}",
            "Control Description": fake.sentence(nb_words=10),
            "Control Owner": fake.name(),
            "Implementation Date": fake.date_between(start_date='-2y', end_date='today'),
            "Review Date": fake.date_between(start_date='today', end_date='+1y'),
            "Status": random.choice(["Active", "Inactive", "Under Review"])
        }
        data.append(record)
    return pd.DataFrame(data)

def generate_issues_data(num_records: int) -> pd.DataFrame:
    data = []
    for i in range(num_records):
        record = {
            "Issue ID": f"ISS-{i+1:04d}",
            "Issue Description": fake.sentence(nb_words=12),
            "Severity": random.choice(["Low", "Medium", "High", "Critical"]),
            "Status": random.choice(["Open", "In Progress", "Resolved", "Closed"]),
            "Reporter": fake.name(),
            "Date Reported": fake.date_between(start_date='-1y', end_date='today'),
            "Resolution Date": fake.date_between(start_date='today', end_date='+6m') if random.choice([True, False]) else None
        }
        data.append(record)
    return pd.DataFrame(data)

def generate_loss_data(num_records: int, loss_type: str) -> pd.DataFrame:
    data = []
    for i in range(num_records):
        record = {
            "Loss ID": f"{loss_type.upper()[:3]}-{i+1:04d}",
            "Loss Description": fake.sentence(nb_words=15),
            "Loss Amount": round(random.uniform(1000, 1000000), 2),
            "Currency": random.choice(["USD", "EUR", "GBP"]),
            "Event Date": fake.date_between(start_date='-2y', end_date='today'),
            "Recovery Amount": round(random.uniform(0, 50000), 2),
            "Business Line": random.choice(["Retail Banking", "Investment Banking", "Corporate Banking", "Trading"])
        }
        data.append(record)
    return pd.DataFrame(data)

def generate_orx_scenarios_data(num_records: int) -> pd.DataFrame:
    data = []
    for i in range(num_records):
        record = {
            "Scenario ID": f"ORX-{i+1:04d}",
            "Scenario Description": fake.sentence(nb_words=20),
            "Risk Category": random.choice(["Operational", "Credit", "Market", "Liquidity", "Reputation"]),
            "Probability": random.choice(["Low", "Medium", "High"]),
            "Impact": random.choice(["Low", "Medium", "High", "Severe"]),
            "Mitigation": fake.sentence(nb_words=8),
            "Business Line": random.choice(["Retail Banking", "Investment Banking", "Corporate Banking", "Trading"])
        }
        data.append(record)
    return pd.DataFrame(data)

def generate_data(data_type: str, num_records: int) -> pd.DataFrame:
    if data_type == "controls":
        return generate_controls_data(num_records)
    elif data_type == "issues":
        return generate_issues_data(num_records)
    elif data_type in ["external_loss", "internal_loss"]:
        return generate_loss_data(num_records, data_type)
    elif data_type == "orx_scenarios":
        return generate_orx_scenarios_data(num_records)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

def save_data(df: pd.DataFrame, filename: str, format: str):
    if format == "csv":
        df.to_csv(filename, index=False)
    elif format == "xlsx":
        df.to_excel(filename, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

def main():
    parser = argparse.ArgumentParser(description="Generate mock data for the task ticket platform")
    parser.add_argument("--type", required=True, choices=DATA_TYPE_CONFIGS.keys(),
                        help="Type of data to generate")
    parser.add_argument("--records", type=int, default=50,
                        help="Number of records to generate (default: 50)")
    parser.add_argument("--format", choices=["csv", "xlsx"], default="xlsx",
                        help="Output format (default: xlsx)")
    parser.add_argument("--output-dir", default="test_data",
                        help="Output directory (default: test_data)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = generate_data(args.type, args.records)
    
    filename = f"mock_{args.type}_{args.records}.{args.format}"
    filepath = os.path.join(args.output_dir, filename)
    
    save_data(df, filepath, args.format)
    
    print(f"Generated {args.records} records of {args.type} data")
    print(f"Saved to: {filepath}")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    main()
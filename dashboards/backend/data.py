import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()

# Static category mappings
taxonomies = ["Fraud", "Cybersecurity", "Data Privacy", "Third-Party Risk", "Process Failures"]
business_lines = ["Corporate Finance", "Retail Banking", "Wealth Management", "Asset Management"]
ubs_divisions = ["UBS Europe SE", "UBS AG", "UBS Switzerland", "UBS UK Branch"]
event_categories = ["External Fraud", "Internal Fraud", "Execution Errors", "Compliance Breach"]
regions = ["EMEA", "APAC", "AMERICAS"]
countries = ["Germany", "India", "USA", "UK", "Singapore", "Switzerland"]
activities = ["Trade Booking", "Client Onboarding", "Transaction Settlement", "Risk Assessment"]
themes = ["Model Risk", "Automation Risk", "Human Error", "Outsourcing Risk", "Third-Party AI Risk"]
steps = ["Data Collection", "Preprocessing", "Model Selection", "Evaluation", "Deployment"]

def generate_mock_row():
    settlement_date = fake.date_between(start_date='-3y', end_date='today')
    return {
        "reference_id_code": fake.uuid4(),
        "parent_name": fake.company(),
        "ama_peer_bank": fake.company(),
        "description_of_event": f"{random.choice(taxonomies)} incident occurred in {random.choice(activities)} process.",
        "nfr_taxonomy": random.choice(taxonomies),
        "loss_amount___m_": round(random.uniform(0.1, 100.0), 2),
        "basel_business_line__level1_": random.choice(business_lines),
        "basel_business_line__level_2": random.choice(business_lines),
        "business_unit": fake.bs(),
        "ubs_business_division": random.choice(ubs_divisions),
        "event_risk_category": random.choice(event_categories),
        "sub_risk_category": random.choice(event_categories),
        "activity": random.choice(activities),
        "country_of_incident": random.choice(countries),
        "event_region": random.choice(regions),
        "month__year_of_settlement": settlement_date.strftime("%m/%Y"),
        "multiple_firms_impacted_code": random.choice(["Y", "N"]),
        "single_event_multiple_loss_code": random.choice(["Y", "N"]),
        "date_of_entry": (settlement_date + timedelta(days=random.randint(1, 60))).strftime("%Y-%m-%d"),
        "nfr_taxonomy_number": f"NFR-{random.randint(1000,9999)}",
        "root_cause": random.choice(["Process gap", "System failure", "Insider threat", "Vendor failure"]),
        "learning_outcome": f"The incident highlighted gaps in {random.choice(activities)}.",
        "impact": f"{random.choice(['Severe', 'Moderate', 'Minimal'])} impact on operations and reputational risk.",
        "summary": f"{random.choice(taxonomies)} event causing ${random.randint(1,100)}M loss.",
        "ai_nfr_cluster": random.choice(["AI Ops", "AI Governance", "AI Compliance"]),
        "ai_nfr_taxonomy": random.choice(taxonomies),
        "ai_risk_theme": random.choice(themes),
        "ai_reasoning_taxonomy_steps": ", ".join(random.sample(steps, k=3)),
        "ai_reasoning_risk_theme": f"Risk due to {random.choice(themes)} in {random.choice(activities)}."
    }

# Generate 5000 rows
data = [generate_mock_row() for _ in range(5000)]
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("external_loss_mock_data.csv", index=False)
print("Mock data saved to external_loss_mock_data.csv")
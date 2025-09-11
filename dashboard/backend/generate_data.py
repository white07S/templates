import pandas as pd
import json
import random
from datetime import datetime, timedelta

# Sample data for generating realistic dummy data
AI_TAXONOMIES = [
    "Operational Risk", "Credit Risk", "Market Risk", "Liquidity Risk",
    "Compliance Risk", "Technology Risk", "Model Risk", "Reputational Risk",
    "Strategic Risk", "Legal Risk", "Cyber Risk", "Third Party Risk"
]

ERMS_TAXONOMIES = [
    "OR-001: Processing Error", "OR-002: System Failure", "OR-003: Human Error",
    "CR-001: Default Risk", "CR-002: Concentration Risk", "CR-003: Settlement Risk",
    "MR-001: Interest Rate Risk", "MR-002: FX Risk", "MR-003: Equity Risk",
    "LR-001: Funding Risk", "LR-002: Market Liquidity Risk",
    "TR-001: Infrastructure Risk", "TR-002: Data Risk", "TR-003: Security Risk"
]

DESCRIPTIONS_TEMPLATES = [
    "Failed transaction processing in {system} resulting in {impact}",
    "System outage in {system} affecting {impact}",
    "Data quality issue in {system} leading to {impact}",
    "Security breach detected in {system} with {impact}",
    "Compliance violation in {system} causing {impact}",
    "Process failure in {system} resulting in {impact}",
    "Integration error between {system} and external service",
    "Performance degradation in {system} impacting {impact}",
    "Configuration error in {system} leading to {impact}",
    "Authorization bypass in {system} with {impact}"
]

SYSTEMS = ["Trading Platform", "Risk Engine", "Settlement System", "Client Portal", 
           "Reporting System", "Payment Gateway", "Data Warehouse", "CRM System"]

IMPACTS = ["client disruption", "financial loss", "regulatory concern", "data corruption",
          "service unavailability", "audit findings", "reputation damage", "operational delay"]

def generate_description():
    template = random.choice(DESCRIPTIONS_TEMPLATES)
    system = random.choice(SYSTEMS)
    impact = random.choice(IMPACTS)
    return template.format(system=system, impact=impact)

def generate_json_data(complexity="medium"):
    if complexity == "simple":
        return {
            "status": random.choice(["active", "resolved", "pending"]),
            "severity": random.choice(["low", "medium", "high", "critical"]),
            "created_date": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat()
        }
    elif complexity == "medium":
        return {
            "metadata": {
                "source": random.choice(["system", "manual", "external"]),
                "priority": random.choice(["P1", "P2", "P3", "P4"]),
                "assigned_to": f"user_{random.randint(100, 999)}",
                "tags": random.sample(["urgent", "review", "escalated", "monitoring", "resolved"], k=random.randint(1, 3))
            },
            "financial_impact": {
                "amount": round(random.uniform(1000, 1000000), 2),
                "currency": "USD",
                "estimated": random.choice([True, False])
            },
            "timeline": {
                "reported": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                "acknowledged": (datetime.now() - timedelta(days=random.randint(1, 25))).isoformat(),
                "resolved": (datetime.now() - timedelta(days=random.randint(1, 20))).isoformat() if random.choice([True, False]) else None
            }
        }
    else:  # complex
        return {
            "detailed_analysis": {
                "root_cause_category": random.choice(["human_error", "system_failure", "process_gap", "external_factor"]),
                "contributing_factors": random.sample([
                    "inadequate_training", "system_overload", "communication_breakdown", 
                    "process_deviation", "technical_debt", "vendor_issue"
                ], k=random.randint(1, 3)),
                "risk_indicators": {
                    "probability": random.uniform(0.1, 1.0),
                    "impact_score": random.randint(1, 10),
                    "risk_rating": random.choice(["Low", "Medium", "High", "Very High"])
                }
            },
            "remediation": {
                "immediate_actions": [
                    f"Action {i+1}: {random.choice(['System restart', 'Manual override', 'Data correction', 'Process bypass'])}"
                    for i in range(random.randint(1, 3))
                ],
                "long_term_solutions": [
                    f"Solution {i+1}: {random.choice(['System upgrade', 'Process redesign', 'Training program', 'Control enhancement'])}"
                    for i in range(random.randint(1, 2))
                ]
            },
            "stakeholders": {
                "business_owner": f"manager_{random.randint(10, 99)}",
                "technical_lead": f"tech_{random.randint(10, 99)}",
                "risk_reviewer": f"risk_{random.randint(10, 99)}"
            }
        }

def generate_dataset(dataset_type, num_records=1000):
    records = []
    
    for i in range(num_records):
        record = {
            "id": i + 1,
            "dataset_type": dataset_type,
            "description": generate_description(),
            "ai_taxonomy": random.choice(AI_TAXONOMIES),
            "current_erms_taxonomy": random.choice(ERMS_TAXONOMIES),
            "raw_meta_data": json.dumps(generate_json_data("medium")),
            "ai_taxonomy_detail": json.dumps(generate_json_data("simple")),
            "ai_root_cause": json.dumps(generate_json_data("complex")),
            "ai_enrichment": json.dumps(generate_json_data("medium"))
        }
        records.append(record)
    
    return pd.DataFrame(records)

def main():
    datasets = ["external_loss", "internal_loss", "issues", "controls"]
    
    # Generate different sized datasets to test scalability
    dataset_sizes = {
        "external_loss": 5000,    # Large dataset
        "internal_loss": 3000,    # Medium dataset  
        "issues": 10000,          # Very large dataset
        "controls": 1500          # Small dataset
    }
    
    all_data = pd.DataFrame()
    
    for dataset_type in datasets:
        print(f"Generating {dataset_sizes[dataset_type]} records for {dataset_type}...")
        df = generate_dataset(dataset_type, dataset_sizes[dataset_type])
        all_data = pd.concat([all_data, df], ignore_index=True)
    
    # Save combined dataset
    all_data.to_csv("/Users/preetam/Develop/dashboard/backend/dashboard_data.csv", index=False)
    
    # Also save individual datasets for reference
    for dataset_type in datasets:
        subset = all_data[all_data["dataset_type"] == dataset_type]
        subset.to_csv(f"/Users/preetam/Develop/dashboard/backend/{dataset_type}_data.csv", index=False)
    
    print(f"Generated {len(all_data)} total records across all datasets")
    print("Files created:")
    print("- dashboard_data.csv (combined)")
    for dataset_type in datasets:
        print(f"- {dataset_type}_data.csv")

if __name__ == "__main__":
    main()
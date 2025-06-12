#!/usr/bin/env python3
"""
UBS Loss Data Analysis Agents using Agno Framework
Agents that write and execute Python code for data analysis and visualization
"""

from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.tools.python import PythonTools
import yaml
import argparse

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

UBS_COLORS = {
    "ubs-black": "#000000",
    "ubs-red": "#e60000",
    "ubs-white": "#ffffff",
    "ubs-red-web": "#da0000",
    "ubs-bordeaux1": "#bd000c",
    "ubs-bordeaux50": "#b03974",
    "ubs-sand": "#cfbd9b",
    "ubs-caramel": "#cfbd9b",
    "ubs-ginger": "#e05bd0",
    "ubs-chocolate": "#4d3c2f",
    "ubs-clay": "#7b6b59",
    "ubs-mouse": "#beb29e",
    "ubs-curry": "#e5b01c",
    "ubs-amber-web": "#f2c551",
    "ubs-warm5": "#5b5e5d",
    "ubs-honey": "#edc860",
    "ubs-straw": "#f2d88e",
    "ubs-chestnut-web": "#ba0000",
    "ubs-chestnut": "#a43725",
    "ubs-terracotta": "#c07156",
    "ubs-cinnamon": "#e6b64d"
}

class LossDataAnalyst(Agent):
    """Agent specialized in writing and executing code for loss data analysis"""
    
    def __init__(self):
        super().__init__(
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=[PythonTools()],
            instructions=f"""You are a UBS External Loss Data Analyst agent. You write and execute Python code to analyze operational risk data.

DATA SCHEMA:
The data file 'external_loss_data.csv' contains these columns:
{yaml.dump(config['data_schema']['columns'], default_flow_style=False)}

ANALYSIS CAPABILITIES:
- Load and preprocess loss data from CSV
- Calculate risk metrics (VaR, Expected Shortfall, etc.)
- Analyze trends by time, business line, risk category
- Generate statistical summaries and insights
- Identify patterns and outliers

IMPORTANT INSTRUCTIONS:
1. ALWAYS start by loading the data with pandas
2. Handle data cleaning (dates, numeric conversions, etc.)
3. Write complete, executable Python code for analysis
4. Provide clear insights and interpretations
5. Include error handling and data validation

EXAMPLE DATA PREPROCESSING:
```python
import pandas as pd
import numpy as np
from datetime import datetime

# Load data
df = pd.read_csv('external_loss_data.csv')

# Clean data
df['loss_amount_m'] = pd.to_numeric(df['loss_amount___m_'], errors='coerce')
df['date_of_entry'] = pd.to_datetime(df['date_of_entry'])
df['settlement_date'] = pd.to_datetime(df['month__year_of_settlement'], format='%m/%Y', errors='coerce')
```

When asked questions, write Python code to analyze the data and provide insights.""",
            show_tool_calls=True,
            markdown=True
        )

class LossDataVisualizer(Agent):
    """Agent specialized in creating UBS-branded visualizations"""

    def __init__(self):
        super().__init__(
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=[PythonTools()],
            instructions=f"""You are a UBS Data Visualization Agent. You write and execute Python code to create professional, UBS-branded charts and visualizations.

UBS BRAND COLORS (use these hex codes):
{yaml.dump(UBS_COLORS, default_flow_style=False)}

DATA SCHEMA:
{yaml.dump(config['data_schema']['columns'], default_flow_style=False)}

VISUALIZATION GUIDELINES:
- ALWAYS use UBS brand colors (default to ubs-red #e60000)
- Create professional, clean visualizations
- Include proper titles, labels, and legends
- Use appropriate chart types for data relationships
- Apply UBS styling consistently

STANDARD SETUP CODE:
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# UBS Colors
UBS_COLORS = {str(UBS_COLORS)}

# Load and prepare data
df = pd.read_csv('external_loss_data.csv')
df['loss_amount_m'] = pd.to_numeric(df['loss_amount___m_'], errors='coerce')
df['settlement_date'] = pd.to_datetime(df['month__year_of_settlement'], format='%m/%Y', errors='coerce')

# Set UBS styling
plt.style.use('seaborn-v0_8-whitegrid')
```

CHART TYPE RECOMMENDATIONS:
- Time series: Line charts with UBS red
- Categories: Bar charts with UBS color palette
- Distributions: Histograms with UBS red
- Correlations: Scatter plots with UBS colors
- Comparisons: Multi-colored bar/pie charts

Always write complete, executable code that produces the visualization.""",
            show_tool_calls=True,
            markdown=True
        )

class LossDataReporter(Agent):
    """Agent specialized in generating comprehensive risk reports"""

    def __init__(self):
        super().__init__(
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=[PythonTools()],
            instructions=f"""You are a UBS Risk Reporting Agent. You write Python code to generate comprehensive operational risk reports.

DATA SCHEMA:
{yaml.dump(config['data_schema']['columns'], default_flow_style=False)}

REPORTING CAPABILITIES:
- Executive summaries with key metrics
- Risk analysis by business line, geography, category
- Trend analysis and forecasting
- Regulatory reporting metrics
- Root cause analysis
- Impact assessments

REPORT STRUCTURE:
- Executive Summary (key metrics, total losses)
- Risk Metrics (VaR, Expected Shortfall, volatility)
- Business Line Analysis
- Geographic Distribution
- Root Cause Analysis
- Trend Analysis
- Recommendations

EXAMPLE RISK CALCULATIONS:
```python
# VaR and Expected Shortfall
losses = df['loss_amount_m'].dropna()
var_95 = np.percentile(losses, 95)
var_99 = np.percentile(losses, 99)
es_95 = losses[losses >= var_95].mean()
es_99 = losses[losses >= var_99].mean()
```

Write Python code to generate structured, professional reports with quantitative analysis.""",
            show_tool_calls=True,
            markdown=True
        )

def main():
    """Main CLI interface for UBS Loss Data Analysis Agents"""

    parser = argparse.ArgumentParser(description='UBS Loss Data Analysis Agents')
    parser.add_argument('--mode', choices=['analyze', 'visualize', 'report'], 
                       required=True, help='Analysis mode')
    parser.add_argument('--query', type=str, required=True, 
                       help='Natural language query')

    args = parser.parse_args()

    # Initialize appropriate agent
    if args.mode == 'analyze':
        agent = LossDataAnalyst()
        print("🔍 UBS Loss Data Analyst Agent")
    elif args.mode == 'visualize':
        agent = LossDataVisualizer()
        print("📊 UBS Data Visualization Agent")
    elif args.mode == 'report':
        agent = LossDataReporter()
        print("📋 UBS Risk Reporting Agent")

    print(f"Query: {args.query}")
    print("=" * 60)

    # Execute agent
    try:
        response: RunResponse = agent.run(args.query)
        print(response.content)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
import json
from pydantic import BaseModel
from typing import Dict, Any


class SummaryResponse(BaseModel):
    summary: str
    key_points: list[str]
    risk_level: str
    impact_assessment: str


def render_summary_prompt(data_case: str, row: Dict[str, Any]) -> str:
    config_descriptions = {
        "external_loss": "This is a loss description outside of UBS company",
        "internal_loss": "This is a loss description within UBS company"
    }
    
    prompt = f"""
You are a senior risk analyst at UBS specializing in operational risk assessment and non-financial risk taxonomy.

# Task
Your task is to analyze and summarize loss event data. {config_descriptions.get(data_case, "Process loss event data")}.

# Data Points
You will analyze the following data points:

1. DescriptionOfEvent: {row.get('DescriptionOfEvent', 'N/A')}
2. nfr_taxonomy: {row.get('nfr_taxonomy', 'N/A')}

# Instructions
<chain-of-thought>
1. First, carefully read and understand the event description
2. Identify the key risk factors and impact areas
3. Assess the severity and implications based on the NFR taxonomy
4. Determine the risk level (Low, Medium, High, Critical)
5. Provide actionable insights and impact assessment
</chain-of-thought>

# Output Format

Please provide your response in the following Pydantic schema format:

```python
class SummaryResponse(BaseModel):
    summary: str  # A concise summary of the event (2-3 sentences)
    key_points: list[str]  # 3-5 key points extracted from the event
    risk_level: str  # One of: Low, Medium, High, Critical
    impact_assessment: str  # Assessment of potential impact and recommendations
```

Your response should be a valid JSON object matching this schema:

```json
{{
    "summary": "Brief summary of the loss event...",
    "key_points": ["Point 1", "Point 2", "Point 3"],
    "risk_level": "Medium",
    "impact_assessment": "Detailed assessment and recommendations..."
}}
```

Requirements:
- Summary should be concise but comprehensive
- Key points should highlight the most important aspects
- Risk level should reflect the severity based on available information
- Impact assessment should include potential consequences and recommended actions
- Ensure all fields are properly filled based on the provided data

Provide only the JSON response without any additional text or formatting.
"""
    
    return prompt
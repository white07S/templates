#!/usr/bin/env python3
"""
UBS Loss Data Analysis Chatbot using FastAPI and Agno Framework
A web-based chatbot interface for data analysis and visualization
FIXED VERSION - Resolves agent loops and code execution issues
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import uuid
import asyncio
import pandas as pd
import yaml
from datetime import datetime
import logging
import traceback
import time

from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.tools.python import PythonTools
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="UBS Loss Data Analysis Chatbot", version="1.0.0")

# Create directories
os.makedirs("static/plots", exist_ok=True)
os.makedirs("static/data", exist_ok=True)
os.makedirs("tmp", exist_ok=True)

# Global variables for data and configuration
loss_data_df = None
config = None

# Load configuration
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    # Default configuration if file doesn't exist
    config = {
        'data_schema': {
            'columns': {
                'loss_amount___m_': 'Loss amount in millions USD',
                'date_of_entry': 'Date when loss was entered',
                'month__year_of_settlement': 'Settlement date (MM/YYYY format)',
                'business_line': 'Business line where loss occurred',
                'risk_category': 'Operational risk category',
                'geography': 'Geographic region',
                'root_cause': 'Primary cause of the loss'
            }
        },
        'ubs_colors': {
            'ubs-red': '#e60000',
            'ubs-blue': '#0073e6',
            'ubs-gray': '#767676',
            'ubs-light-gray': '#f5f5f5',
            'ubs-dark-gray': '#333333'
        }
    }

# Request/Response models
class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: Optional[str] = None
    plots: Optional[List[str]] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    agent_type: str = "analyze"  # 'analyze', 'visualize', 'report'

class ChatResponse(BaseModel):
    response: str
    session_id: str
    plots: List[str] = []
    agent_type: str

# Session storage
sessions: Dict[str, Dict[str, Any]] = {}

def load_data_in_memory():
    """Load the external loss data into memory"""
    global loss_data_df
    try:
        if os.path.exists('external_loss_data.csv'):
            loss_data_df = pd.read_csv('external_loss_data.csv')
            # Basic preprocessing
            loss_data_df['loss_amount_m'] = pd.to_numeric(loss_data_df.get('loss_amount___m_', []), errors='coerce')
            loss_data_df['date_of_entry'] = pd.to_datetime(loss_data_df.get('date_of_entry', []), errors='coerce')
            loss_data_df['settlement_date'] = pd.to_datetime(
                loss_data_df.get('month__year_of_settlement', []), 
                format='%m/%Y', 
                errors='coerce'
            )
            logger.info(f"Loaded {len(loss_data_df)} records into memory")
            
            # Save processed data for agents to use
            loss_data_df.to_csv('static/data/processed_loss_data.csv', index=False)
        else:
            logger.warning("external_loss_data.csv not found. Creating sample data.")
            # Create sample data for demonstration
            import numpy as np
            np.random.seed(42)
            n_records = 1000
            
            # Create realistic sample data
            sample_data = {
                'loss_amount___m_': np.random.lognormal(0, 1.5, n_records),
                'date_of_entry': pd.date_range('2020-01-01', periods=n_records, freq='D'),
                'month__year_of_settlement': [f"{(i%12)+1:02d}/202{(i//365)+0}" for i in range(n_records)],
                'business_line': np.random.choice(['Investment Bank', 'Wealth Management', 'Asset Management', 'Personal Banking'], n_records),
                'risk_category': np.random.choice(['Fraud', 'Technology', 'Process', 'Human Error', 'External'], n_records),
                'geography': np.random.choice(['Americas', 'EMEA', 'APAC'], n_records),
                'root_cause': np.random.choice(['System Failure', 'Human Error', 'External Event', 'Process Breakdown', 'Fraud'], n_records)
            }
            
            loss_data_df = pd.DataFrame(sample_data)
            loss_data_df['loss_amount_m'] = loss_data_df['loss_amount___m_']
            loss_data_df.to_csv('static/data/processed_loss_data.csv', index=False)
            logger.info("Created sample data for demonstration")
            
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        loss_data_df = pd.DataFrame()

class CustomPythonTools(PythonTools):
    """Custom PythonTools with better error handling and plot management"""
    
    def __init__(self):
        super().__init__()
        self.max_retries = 2
        self.current_retry = 0
    
    def run_python_code(self, code: str, **kwargs) -> str:
        """Override to add better error handling"""
        try:
            # Clean the code to prevent syntax errors
            cleaned_code = self._clean_code(code)
            
            # Try to run the cleaned code
            result = super().run_python_code(cleaned_code, **kwargs)
            self.current_retry = 0  # Reset retry counter on success
            return result
            
        except SyntaxError as e:
            logger.error(f"Syntax error in Python code: {str(e)}")
            if self.current_retry < self.max_retries:
                self.current_retry += 1
                # Try to fix common syntax issues
                fixed_code = self._fix_common_syntax_errors(code)
                return self.run_python_code(fixed_code, **kwargs)
            else:
                return f"Syntax error after {self.max_retries} attempts: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error executing Python code: {str(e)}")
            return f"Execution error: {str(e)}"
    
    def _clean_code(self, code: str) -> str:
        """Clean code to prevent common issues"""
        # Remove any problematic characters
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove any invisible characters that might cause issues
        code = ''.join(char for char in code if ord(char) >= 32 or char in '\n\t')
        
        # Fix line continuation issues
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove trailing backslashes that might cause issues
            if line.endswith('\\') and not line.endswith('\\\\'):
                line = line[:-1].rstrip()
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _fix_common_syntax_errors(self, code: str) -> str:
        """Attempt to fix common syntax errors"""
        # Remove problematic line continuations
        code = code.replace('\\\n', ' ')
        code = code.replace('\\', '')
        
        # Fix common quote issues
        code = code.replace('"', '"').replace('"', '"')
        code = code.replace(''', "'").replace(''', "'")
        
        return code

class LossDataAnalyst(Agent):
    """Agent specialized in writing and executing code for loss data analysis"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        # Create memory database for this agent
        memory_db = SqliteMemoryDb(
            table_name="analyst_memories", 
            db_file=f"tmp/memory_{session_id}.db"
        )
        memory = Memory(db=memory_db)
        
        super().__init__(
            model=OpenAIChat(id="o3-mini"),
            tools=[CustomPythonTools()],
            memory=memory,
            enable_agentic_memory=True,
            enable_user_memories=True,
            add_history_to_messages=True,
            num_history_runs=3,  # Reduced to prevent too much context
            instructions=f"""You are a UBS External Loss Data Analyst agent. You write and execute Python code to analyze operational risk data.

CRITICAL INSTRUCTIONS FOR CODE GENERATION:

Data example:
reference_id_code,parent_name,ama_peer_bank,description_of_event,nfr_taxonomy,loss_amount___m_,basel_business_line__level1_,basel_business_line__level_2,business_unit,ubs_business_division,event_risk_category,sub_risk_category,activity,country_of_incident,event_region,month__year_of_settlement,multiple_firms_impacted_code,single_event_multiple_loss_code,date_of_entry,nfr_taxonomy_number,root_cause,learning_outcome,impact,summary,ai_nfr_cluster,ai_nfr_taxonomy,ai_risk_theme,ai_reasoning_taxonomy_steps,ai_reasoning_risk_theme,settlement_year,settlement_month
fe4123ef-2509-4e47-88e9-57ec33933004,"Nguyen, Mccarty and Mann","James, Williams and Downs",Fraud incident occurred in Risk Assessment process.,Cybersecurity,48.14,Asset Management,Wealth Management,cultivate world-class communities,UBS Europe SE,Compliance Breach,Execution Errors,Client Onboarding,India,EMEA,2023-11-01,N,Y,2023-12-02,NFR-1764,Insider threat,The incident highlighted gaps in Transaction Settlement.,Moderate impact on operations and reputational risk.,Fraud event causing $74M loss.,AI Ops,Fraud,Outsourcing Risk,"Data Collection, Preprocessing, Model Selection",Risk due to Human Error in Client Onboarding.,2023,11

1. ALWAYS write clean, simple Python code with no line continuation characters (\\)
2. Use simple variable names and avoid complex string formatting
3. If code fails with syntax error, COMPLETELY REWRITE it with a different approach
4. NEVER retry the same failing code pattern
5. Keep code blocks short and focused

DATA LOCATION: 'static/data/processed_loss_data.csv' ({len(loss_data_df) if loss_data_df is not None else 0} records)

STANDARD ANALYSIS TEMPLATE:
```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('static/data/processed_loss_data.csv')
print(f"Loaded {{len(df)}} records")

# Your analysis here
# Example: Basic statistics
print("Loss Amount Statistics:")
print(df['loss_amount_m'].describe())
```

RISK CALCULATIONS EXAMPLE:
```python
# Calculate VaR and Expected Shortfall
losses = df['loss_amount_m'].dropna()
var_95 = np.percentile(losses, 95)
var_99 = np.percentile(losses, 99)
es_95 = losses[losses >= var_95].mean()
print(f"VaR 95%: ${{var_95:.2f}}M")
print(f"Expected Shortfall 95%: ${{es_95:.2f}}M")
```

DATA SCHEMA:
{yaml.dump(config['data_schema']['columns'], default_flow_style=False)}

If you encounter ANY syntax error, immediately try a completely different approach rather than fixing the same code.""",
            show_tool_calls=True,
            markdown=True
        )

class LossDataVisualizer(Agent):
    """Agent specialized in creating UBS-branded visualizations"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        # Create memory database for this agent
        memory_db = SqliteMemoryDb(
            table_name="visualizer_memories", 
            db_file=f"tmp/memory_{session_id}.db"
        )
        memory = Memory(db=memory_db)
        
        super().__init__(
            model=OpenAIChat(id="o3-mini"),
            tools=[CustomPythonTools()],
            memory=memory,
            enable_agentic_memory=True,
            enable_user_memories=True,
            add_history_to_messages=True,
            num_history_runs=3,
            instructions=f"""You are a UBS Data Visualization Agent. Create professional UBS-branded charts.

CRITICAL PLOT SAVING REQUIREMENTS:
"always use: import matplotlib matplotlib.use('Agg')  # Use non-interactive backend"

Data example:
reference_id_code,parent_name,ama_peer_bank,description_of_event,nfr_taxonomy,loss_amount___m_,basel_business_line__level1_,basel_business_line__level_2,business_unit,ubs_business_division,event_risk_category,sub_risk_category,activity,country_of_incident,event_region,month__year_of_settlement,multiple_firms_impacted_code,single_event_multiple_loss_code,date_of_entry,nfr_taxonomy_number,root_cause,learning_outcome,impact,summary,ai_nfr_cluster,ai_nfr_taxonomy,ai_risk_theme,ai_reasoning_taxonomy_steps,ai_reasoning_risk_theme,settlement_year,settlement_month
fe4123ef-2509-4e47-88e9-57ec33933004,"Nguyen, Mccarty and Mann","James, Williams and Downs",Fraud incident occurred in Risk Assessment process.,Cybersecurity,48.14,Asset Management,Wealth Management,cultivate world-class communities,UBS Europe SE,Compliance Breach,Execution Errors,Client Onboarding,India,EMEA,2023-11-01,N,Y,2023-12-02,NFR-1764,Insider threat,The incident highlighted gaps in Transaction Settlement.,Moderate impact on operations and reputational risk.,Fraud event causing $74M loss.,AI Ops,Fraud,Outsourcing Risk,"Data Collection, Preprocessing, Model Selection",Risk due to Human Error in Client Onboarding.,2023,11


1. ALWAYS save plots to 'static/plots/' directory with unique names
2. Use this EXACT template for plot saving:
3. Import time and uuid at the top
4. Use simple, clean code with NO line continuation characters
5. If code fails, try a COMPLETELY different visualization approach

MANDATORY PLOT TEMPLATE:
```python

[!CAUTION, dont do plt.show, just save it]
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import uuid

# UBS Colors
UBS_RED = '#e60000'
UBS_BLUE = '#0073e6'
UBS_GRAY = '#767676'

# Load data
df = pd.read_csv('static/data/processed_loss_data.csv')

# Create plot
plt.figure(figsize=(10, 6))
# Your plot code here
plt.title('Your Title', color=UBS_RED, fontsize=14, fontweight='bold')
plt.xlabel('X Label')
plt.ylabel('Y Label')

# ALWAYS save plot with unique filename
timestamp = int(time.time())
unique_id = str(uuid.uuid4())[:8]
filename = f'plot_{{timestamp}}_{{unique_id}}.png'
plt.savefig(f'static/plots/{{filename}}', bbox_inches='tight', dpi=300)
print(f'Plot saved as: {{filename}}')e
plt.close()
```

SIMPLE VISUALIZATION EXAMPLES:

Bar Chart:
```python
import matplotlib.pyplot as plt
import pandas as pd
import time
import uuid

df = pd.read_csv('static/data/processed_loss_data.csv')
plt.figure(figsize=(10, 6))
data = df.groupby('business_line')['loss_amount_m'].sum()
data.plot(kind='bar', color='#e60000')
plt.title('Losses by Business Line', fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()

filename = f'plot_{{int(time.time())}}_{{str(uuid.uuid4())[:8]}}.png'
plt.savefig(f'static/plots/{{filename}}', bbox_inches='tight', dpi=300)
print(f'Plot saved as: {{filename}}')
plt.close()
```
REMEMBER: If ANY error occurs, try a completely different chart type or approach!""",
            show_tool_calls=True,
            markdown=True
        )

class LossDataReporter(Agent):
    """Agent specialized in generating comprehensive risk reports"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        # Create memory database for this agent
        memory_db = SqliteMemoryDb(
            table_name="reporter_memories", 
            db_file=f"tmp/memory_{session_id}.db"
        )
        memory = Memory(db=memory_db)
        
        super().__init__(
            model=OpenAIChat(id="o3-mini"),
            tools=[CustomPythonTools()],
            memory=memory,
            enable_agentic_memory=True,
            enable_user_memories=True,
            add_history_to_messages=True,
            num_history_runs=3,
            instructions=f"""You are a UBS Risk Reporting Agent. Generate comprehensive operational risk reports.

            Data example:
reference_id_code,parent_name,ama_peer_bank,description_of_event,nfr_taxonomy,loss_amount___m_,basel_business_line__level1_,basel_business_line__level_2,business_unit,ubs_business_division,event_risk_category,sub_risk_category,activity,country_of_incident,event_region,month__year_of_settlement,multiple_firms_impacted_code,single_event_multiple_loss_code,date_of_entry,nfr_taxonomy_number,root_cause,learning_outcome,impact,summary,ai_nfr_cluster,ai_nfr_taxonomy,ai_risk_theme,ai_reasoning_taxonomy_steps,ai_reasoning_risk_theme,settlement_year,settlement_month
fe4123ef-2509-4e47-88e9-57ec33933004,"Nguyen, Mccarty and Mann","James, Williams and Downs",Fraud incident occurred in Risk Assessment process.,Cybersecurity,48.14,Asset Management,Wealth Management,cultivate world-class communities,UBS Europe SE,Compliance Breach,Execution Errors,Client Onboarding,India,EMEA,2023-11-01,N,Y,2023-12-02,NFR-1764,Insider threat,The incident highlighted gaps in Transaction Settlement.,Moderate impact on operations and reputational risk.,Fraud event causing $74M loss.,AI Ops,Fraud,Outsourcing Risk,"Data Collection, Preprocessing, Model Selection",Risk due to Human Error in Client Onboarding.,2023,11


REPORT GENERATION TEMPLATE:
```python
import pandas as pd
import numpy as np
from datetime import datetime

# Load data
df = pd.read_csv('static/data/processed_loss_data.csv')

print("="*60)
print("UBS OPERATIONAL RISK REPORT")
print("="*60)
print(f"Generated: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")
print(f"Total Records: {{len(df):,}}")
print()

# Executive Summary
total_losses = df['loss_amount_m'].sum()
avg_loss = df['loss_amount_m'].mean()
max_loss = df['loss_amount_m'].max()

print("EXECUTIVE SUMMARY:")
print(f"Total Losses: ${{total_losses:,.2f}}M")
print(f"Average Loss: ${{avg_loss:,.2f}}M")
print(f"Maximum Loss: ${{max_loss:,.2f}}M")
print()

# Risk Metrics
losses = df['loss_amount_m'].dropna()
var_95 = np.percentile(losses, 95)
var_99 = np.percentile(losses, 99)

print("RISK METRICS:")
print(f"VaR 95%: ${{var_95:,.2f}}M")
print(f"VaR 99%: ${{var_99:,.2f}}M")
print()

# Business Line Analysis
print("LOSSES BY BUSINESS LINE:")
bl_losses = df.groupby('business_line')['loss_amount_m'].agg(['sum', 'count', 'mean'])
print(bl_losses.round(2))
```

Create comprehensive reports with:
- Executive summaries
- Risk calculations (VaR, Expected Shortfall)
- Business line breakdowns
- Geographic analysis
- Root cause analysis
- Trend identification
- Actionable recommendations

Keep code simple and error-free. If code fails, use a different reporting approach.""",
            show_tool_calls=True,
            markdown=True
        )

def get_agent(agent_type: str, session_id: str) -> Agent:
    """Factory function to create appropriate agent"""
    try:
        if agent_type == "analyze":
            return LossDataAnalyst(session_id)
        elif agent_type == "visualize":
            return LossDataVisualizer(session_id)
        elif agent_type == "report":
            return LossDataReporter(session_id)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        # Return a basic agent as fallback
        return LossDataAnalyst(session_id)

def extract_plot_filenames(response_content: str) -> List[str]:
    """Extract plot filenames from agent response"""
    plots = []
    lines = response_content.split('\n')
    
    for line in lines:
        if 'Plot saved as:' in line:
            # Extract filename after "Plot saved as: "
            parts = line.split('Plot saved as:')
            if len(parts) > 1:
                filename = parts[1].strip()
                if filename.endswith('.png'):
                    plots.append(filename)
    
    # Also check for recently created files in plots directory
    try:
        current_time = time.time()
        for filename in os.listdir('static/plots'):
            if filename.endswith('.png') and filename.startswith('plot_'):
                file_path = os.path.join('static/plots', filename)
                file_time = os.path.getmtime(file_path)
                # If file was created in the last 30 seconds
                if current_time - file_time < 30:
                    plots.append(filename)
    except Exception as e:
        logger.error(f"Error checking for recent plots: {str(e)}")
    
    return list(set(plots))  # Remove duplicates

@app.on_event("startup")
async def startup_event():
    """Load data on startup"""
    load_data_in_memory()

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML page"""
    try:
        with open("index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found</h1>")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint with improved error handling"""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Initialize session if new
        if session_id not in sessions:
            sessions[session_id] = {
                'messages': [],
                'agent_type': request.agent_type,
                'error_count': 0
            }
        
        # Track errors to prevent infinite loops
        if sessions[session_id].get('error_count', 0) > 3:
            return ChatResponse(
                response="I've encountered multiple errors. Please try a different question or restart the session.",
                session_id=session_id,
                plots=[],
                agent_type=request.agent_type
            )
        
        # Update agent type if changed
        sessions[session_id]['agent_type'] = request.agent_type
        
        # Add user message to session
        user_message = ChatMessage(
            role="user",
            content=request.message,
            timestamp=datetime.now().isoformat()
        )
        sessions[session_id]['messages'].append(user_message)
        
        # Create appropriate agent
        agent = get_agent(request.agent_type, session_id)
        
        # Execute agent with timeout
        try:
            response: RunResponse = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, agent.run, request.message
                ),
                timeout=60.0  # 60 second timeout
            )
            
            # Reset error count on success
            sessions[session_id]['error_count'] = 0
            
        except asyncio.TimeoutError:
            sessions[session_id]['error_count'] += 1
            return ChatResponse(
                response="The analysis is taking too long. Please try a simpler request.",
                session_id=session_id,
                plots=[],
                agent_type=request.agent_type
            )
        
        # Extract plot filenames from response
        plots = extract_plot_filenames(response.content)
        
        # Add assistant message to session
        assistant_message = ChatMessage(
            role="assistant",
            content=response.content,
            timestamp=datetime.now().isoformat(),
            plots=plots
        )
        sessions[session_id]['messages'].append(assistant_message)
        
        return ChatResponse(
            response=response.content,
            session_id=session_id,
            plots=plots,
            agent_type=request.agent_type
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Increment error count
        if session_id in sessions:
            sessions[session_id]['error_count'] = sessions[session_id].get('error_count', 0) + 1
        
        return ChatResponse(
            response=f"I encountered an error: {str(e)}. Please try rephrasing your question.",
            session_id=session_id or str(uuid.uuid4()),
            plots=[],
            agent_type=request.agent_type
        )

@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get chat history for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"messages": sessions[session_id]['messages']}

@app.get("/sessions/{session_id}/reset")
async def reset_session(session_id: str):
    """Reset a session to clear errors"""
    if session_id in sessions:
        sessions[session_id]['error_count'] = 0
        sessions[session_id]['messages'] = []
    return {"message": "Session reset successfully"}

@app.get("/data/summary")
async def get_data_summary():
    """Get summary of loaded data"""
    if loss_data_df is None or loss_data_df.empty:
        return {"error": "No data loaded"}
    
    return {
        "total_records": len(loss_data_df),
        "columns": list(loss_data_df.columns),
        "date_range": {
            "start": loss_data_df['date_of_entry'].min().isoformat() if 'date_of_entry' in loss_data_df.columns else None,
            "end": loss_data_df['date_of_entry'].max().isoformat() if 'date_of_entry' in loss_data_df.columns else None
        },
        "total_loss_amount": float(loss_data_df['loss_amount_m'].sum()) if 'loss_amount_m' in loss_data_df.columns else None
    }

@app.get("/plots/{filename}")
async def get_plot(filename: str):
    """Serve plot images"""
    file_path = f"static/plots/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="Plot not found")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        reload_excludes=["static/*", "tmp/*"]  # Exclude static and tmp from reload monitoring
    )
import os
import pandas as pd
import traceback
from datetime import datetime
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
from agno import Agent
from agno.models.azure_openai import AzureOpenAIChat
from agno.tools.python import PythonTools
import json

# Configuration
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0
VERBOSE = True
MAX_ITERATIONS = 15
MAX_RETRIES = 3

# UBS Color Palette
UBS_COLORS = {
    # PRIMARY COLORS PALETTE
    'ubs-black': '#000000',
    'ubs-red': '#e60000',
    'ubs-white': '#ffffff',
    # SECONDARY PREFERRED
    'ubs-red-web': '#da0000',
    'ubs-bordeaux1': '#bd000c',
    'ubs-bordeaux50': '#b03974',
    'ubs-sand': '#cfbd9b',
    'ubs-caramel': '#cfbd9b',
    'ubs-ginger': '#e05bd0',
    'ubs-chocolate': '#4d3c2f',
    'ubs-clay': '#7b6b59',
    'ubs-mouse': '#beb29e',
    'ubs-curry': '#e5b01c',
    'ubs-amber-web': '#f2c551',
    'ubs-warm5': '#5b5e5d',
    'ubs-honey': '#edc860',
    'ubs-straw': '#f2d88e',
    'ubs-chestnut-web': '#ba0000',
    'ubs-chestnut': '#a43725',
    'ubs-terracotta': '#c07156',
    'ubs-cinnamon': '#e6b64d'
}

class UBSDataAnalyzer:
    """UBS Operational Loss Data Analyzer using Agno and Azure AI"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.processed_df = None
        self.agent = None
        self.setup_data()
        self.setup_agent()
    
    def setup_data(self):
        """Load and preprocess the CSV data"""
        try:
            print(f"Loading data from: {self.csv_path}")
            self.df = pd.read_csv(self.csv_path)
            self.processed_df = self.handle_datetime_errors(self.df.copy())
            print(f"Data loaded successfully. Shape: {self.processed_df.shape}")
            print(f"Columns: {list(self.processed_df.columns)}")
        except Exception as e:
            raise Exception(f"Failed to load data: {str(e)}")
    
    def handle_datetime_errors(self, df):
        """Fix common datetime issues in the DataFrame"""
        # Convert date columns
        date_columns = ['month__year_of_settlement', 'date_of_entry']
        for col in date_columns:
            if col in df.columns:
                try:
                    if df[col].dtype == 'object':
                        if col == 'month__year_of_settlement':
                            df[col] = pd.to_datetime(df[col], format='%m/%Y', errors='coerce')
                        else:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception:
                    pass
        
        # Create date components for easier analysis
        if 'month__year_of_settlement' in df.columns:
            df['settlement_year'] = df['month__year_of_settlement'].dt.year
            df['settlement_month'] = df['month__year_of_settlement'].dt.month
        
        return df
    
    def setup_agent(self):
        """Setup Agno agent with Azure AI and Python tools"""
        try:
            # Setup Azure OpenAI model
            model = AzureOpenAIChat(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                api_version="2024-06-01",
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                api_key=os.environ.get("AZURE_OPENAI_API_KEY")
            )
            
            # Create comprehensive system instructions
            system_instructions = self.create_system_instructions()
            
            # Setup Python tools with data context
            python_tools = PythonTools(
                pip_install=True,
                run_code=True,
                list_files=True,
                read_file=True,
                write_file=True
            )
            
            # Create agent
            self.agent = Agent(
                model=model,
                tools=[python_tools],
                instructions=system_instructions,
                show_tool_calls=VERBOSE,
                max_loops=MAX_ITERATIONS,
                debug_mode=VERBOSE
            )
            
            # Initialize data context in agent
            self.initialize_data_context()
            
        except Exception as e:
            raise Exception(f"Failed to setup agent: {str(e)}")
    
    def create_system_instructions(self):
        """Create comprehensive system instructions for the agent"""
        instructions = f"""
You are an expert UBS operational risk data analyst with access to operational loss data.

AVAILABLE DATA:
- Dataset shape: {self.processed_df.shape if self.processed_df is not None else 'Not loaded'}
- Columns: {list(self.processed_df.columns) if self.processed_df is not None else 'Not loaded'}

UBS COLOR PALETTE (use these for all visualizations):
{json.dumps(UBS_COLORS, indent=2)}

CORE RESPONSIBILITIES:
1. Analyze operational loss data patterns and trends
2. Create professional visualizations using UBS brand colors
3. Provide actionable risk management insights
4. Identify regulatory compliance implications

VISUALIZATION REQUIREMENTS:
- ALWAYS use UBS color palette from the provided colors
- Primary colors (ubs-red, ubs-black) for main elements
- Secondary colors for accents and supporting elements
- Set figure size to (12, 8) for optimal display
- Use high DPI (150) for crisp images
- NEVER use plt.show() - always save plots using plt.savefig()
- ALWAYS close plots after saving using plt.close()
- Include [PLOT_SAVED: filename] marker in responses
- Use bbox_inches='tight' and dpi=150 for best quality

DATA HANDLING BEST PRACTICES:
- The data is already preprocessed with datetime conversions
- 'month__year_of_settlement' is converted to datetime format
- 'settlement_year' and 'settlement_month' are derived columns
- When analyzing loss amounts, consider both absolute and relative impacts
- Look for patterns across time, business lines, and geographic regions
- Pay attention to AI-generated vs traditional risk classifications

DATETIME HANDLING:
- Use 'settlement_year' and 'settlement_month' for time-based analysis
- For date filtering, use pandas datetime comparison methods
- Ensure proper chronological ordering for time series

ERROR HANDLING:
- Always check data types before mathematical operations
- Use .reset_index() after groupby operations
- Handle missing values appropriately
- Verify column names exist before accessing

The operational loss data is available as a pandas DataFrame called 'df' in your Python environment.
"""
        return instructions
    
    def initialize_data_context(self):
        """Initialize the data context in the agent's Python environment"""
        setup_code = f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# UBS Color Palette
UBS_COLORS = {UBS_COLORS}

# Load the preprocessed data
df = pd.read_csv('{self.csv_path}')

# Apply datetime preprocessing
date_columns = ['month__year_of_settlement', 'date_of_entry']
for col in date_columns:
    if col in df.columns:
        try:
            if df[col].dtype == 'object':
                if col == 'month__year_of_settlement':
                    df[col] = pd.to_datetime(df[col], format='%m/%Y', errors='coerce')
                else:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        except Exception:
            pass

# Create date components
if 'month__year_of_settlement' in df.columns:
    df['settlement_year'] = df['month__year_of_settlement'].dt.year
    df['settlement_month'] = df['month__year_of_settlement'].dt.month

print(f"Data loaded successfully. Shape: {{df.shape}}")
print(f"Columns: {{list(df.columns)}}")
print(f"Data types:\\n{{df.dtypes}}")
"""
        
        try:
            response = self.agent.run(f"Execute this setup code to initialize the data environment:\n\n{setup_code}")
            print("Data context initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize data context: {str(e)}")
    
    def analyze(self, query: str) -> str:
        """Run analysis query with retry mechanism"""
        for attempt in range(MAX_RETRIES):
            try:
                print(f"\n[Analysis Attempt {attempt + 1}] Processing query...")
                response = self.agent.run(query)
                return response.content if hasattr(response, 'content') else str(response)
                
            except Exception as e:
                error_trace = traceback.format_exc()
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < MAX_RETRIES - 1:
                    # Provide error context for retry
                    error_context = f"""
Previous attempt failed with error: {error_trace}

Please correct your approach by:
1. Checking datetime conversions and operations
2. Verifying column names and data types  
3. Using only numeric columns for calculations
4. Adding .reset_index() after groupby operations
5. Ensuring plots are saved, not displayed

Re-attempting query: {query}
"""
                    try:
                        response = self.agent.run(error_context)
                        return response.content if hasattr(response, 'content') else str(response)
                    except Exception as retry_error:
                        print(f"Retry also failed: {str(retry_error)}")
                        continue
                else:
                    return f"Maximum retries reached. Error: {str(e)}\n\nPlease rephrase your query or check the data format."
        
        return "Analysis failed after all retries."

def main():
    """Main function to run the UBS Data Analyzer"""
    
    # Check for required environment variables
    required_env_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Configuration
    csv_file_path = "external_loss_mock_data.csv"  # Direct file path
    
    # Check if file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        print("Please ensure the data file exists in the current directory.")
        return
    
    try:
        # Initialize analyzer
        print("Initializing UBS Data Analyzer with Agno and Azure AI...")
        analyzer = UBSDataAnalyzer(csv_file_path)
        
        print("\n" + "="*60)
        print(f"UBS Data Analysis Agent Initialized")
        print(f"Model: {MODEL_NAME} (Azure AI)")
        print(f"Data: {csv_file_path}")
        print("Type 'exit' to quit")
        print("="*60)
        
        # Interactive session
        while True:
            try:
                query = input("\n[UBS Data Query]> ").strip()
                
                if query.lower() in ["exit", "quit", "q"]:
                    print("Session terminated. Goodbye!")
                    break
                    
                if not query:
                    continue
                
                # Process query
                print(f"\n[Processing] {query}")
                response = analyzer.analyze(query)
                print(f"\n[Response]\n{response}")
                
            except KeyboardInterrupt:
                print("\n\nSession terminated by user")
                break
            except Exception as e:
                print(f"\nError processing query: {str(e)}")
                
    except Exception as e:
        print(f"Failed to initialize analyzer: {str(e)}")
        return

if __name__ == "__main__":
    main()

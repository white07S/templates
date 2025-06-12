import os
import argparse
import pandas as pd
import traceback
from datetime import datetime
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI

# Configuration
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0
VERBOSE = True
MAX_ITERATIONS = 15
MAX_RETRIES = 3  # Max retries for the same query

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
    'ubs-cinnamon': '#e6b64'
}

def create_agent(file_path: str) -> AgentType:
    """Create CSV agent with UBS color context and enhanced error handling"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            openai_api_key=os.environ["OPENAI_API_KEY"]
        )
        
        # Create comprehensive context instructions
        context = (
            "CONTEXT INSTRUCTIONS:\n"
            "1. COLOR USAGE: When creating visualizations, use the UBS color palette. "
            "Primary colors should be used for main elements, secondary for accents. "
            "Here are the UBS color definitions in HEX format:\n"
            + "\n".join([f"{name}: {color}" for name, color in UBS_COLORS.items()]) + "\n\n"
            
            "2. DATETIME HANDLING: The 'month__year_of_settlement' column is in 'MM/YYYY' format. "
            "When processing dates:\n"
            "   - Convert using: pd.to_datetime(df['month__year_of_settlement'], format='%m/%Y')\n"
            "   - Avoid mathematical operations on datetime objects directly\n"
            "   - To filter by date, use: df[df['date_column'] > pd.Timestamp('2023-01-01')]\n"
            "   - When grouping by date, extract components first: df['year'] = df['date_column'].dt.year\n\n"
            
            "3. ERROR HANDLING: If you encounter an error, analyze the traceback and:\n"
            "   - Check column data types using df.dtypes\n"
            "   - Verify datetime conversions\n"
            "   - Ensure only numeric columns are used in mathematical operations\n"
            "   - Use .reset_index() after groupby operations\n"
        )
        
        return create_csv_agent(
            llm,
            file_path,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=VERBOSE,
            max_iterations=MAX_ITERATIONS,
            early_stopping_method="generate",
            prefix=context,
            extra_tools=[],
            handle_parsing_errors=True , # Better handle function calling errors,
            allow_dangerous_code=True
        )
    except Exception as e:
        print(f"Agent creation failed: {str(e)}")
        exit(1)

def handle_datetime_errors(df):
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

def main():
    parser = argparse.ArgumentParser(
        description="UBS Operational Loss Data Analysis CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "csv_path",
        help="Path to operational loss data CSV file"
    )
    args = parser.parse_args()

    # Preprocess data to fix common issues
    try:
        df = pd.read_csv(args.csv_path)
        df = handle_datetime_errors(df)
        processed_path = f"processed_{os.path.basename(args.csv_path)}"
        df.to_csv(processed_path, index=False)
        print(f"Preprocessed data saved to: {processed_path}")
    except Exception as e:
        print(f"Data preprocessing failed: {str(e)}")
        processed_path = args.csv_path  # Use original if processing fails

    # Initialize agent with preprocessed data
    agent = create_agent(processed_path)
    print("\n" + "="*50)
    print(f"UBS Data Analysis Agent Initialized (Model: {MODEL_NAME})")
    print("Type 'exit' to quit\n" + "="*50)
    
    # Interactive session with retry mechanism
    while True:
        try:
            query = input("\n[UBS Data Query]> ").strip()
            if query.lower() in ["exit", "quit"]:
                break
            if not query:
                continue
            
            # Execute with retries
            for attempt in range(MAX_RETRIES):
                try:
                    response = agent.invoke({"input": query})
                    print(f"\n[Response] {response['output']}")
                    break  # Exit retry loop if successful
                except Exception as e:
                    error_trace = traceback.format_exc()
                    print(f"\nAttempt {attempt+1} failed: {str(e)}")
                    
                    # Provide detailed error context to agent
                    if attempt < MAX_RETRIES - 1:
                        error_context = (
                            f"Previous attempt failed with error: {error_trace}\n"
                            "Please correct your approach by:\n"
                            "1. Checking datetime conversions and operations\n"
                            "2. Verifying column names and data types\n"
                            "3. Using only numeric columns for calculations\n"
                            "4. Adding .reset_index() after groupby operations\n\n"
                            f"Re-attempting query: {query}"
                        )
                        response = agent.invoke({"input": error_context})
                        print(f"\n[Retry Response] {response['output']}")
                    else:
                        print("\n[Error] Maximum retries reached. Please rephrase your query.")
            
        except KeyboardInterrupt:
            print("\nSession terminated by user")
            break
        except Exception as e:
            print(f"\nFatal error: {str(e)}")

if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")
    
    main()
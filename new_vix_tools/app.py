import os
import json
import uuid
import shutil
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.python import PythonTools
from agno.memory.v2.memory import Memory
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.storage.sqlite import SqliteStorage

# Import the official VisualizationTools or create a proper custom one
try:
    from agno.tools.visualization import VisualizationTools
except ImportError:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    from agno.tools import tool
    
    class VisualizationTools:
        def __init__(self, output_dir: str = "charts"):
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True)
        
        @tool(
            name="create_chart",
            description="Create various types of charts and save them as PNG files",
            show_result=False  # Don't show the result text to avoid extra messages
        )
        def create_chart(
            self, 
            chart_type: str, 
            data_path: str, 
            filename: str, 
            x_column: str = None, 
            y_column: str = None, 
            title: str = "Chart",
            **kwargs
        ) -> str:
            """
            Create a chart from CSV data and save it as a PNG file.
            
            Args:
                chart_type: Type of chart (line, bar, scatter, histogram, etc.)
                data_path: Path to the CSV file
                filename: Name for the output file (without extension)
                x_column: Column name for x-axis
                y_column: Column name for y-axis
                title: Chart title
                **kwargs: Additional chart parameters
            
            Returns:
                str: Success message with filename
            """
            try:
                # Read the data
                df = pd.read_csv(data_path)
                
                # Set up the plot
                plt.figure(figsize=(12, 8))
                plt.style.use('default')
                
                # Create different types of charts
                if chart_type.lower() == 'line':
                    if x_column and y_column:
                        plt.plot(df[x_column], df[y_column], linewidth=2)
                        plt.xlabel(x_column)
                        plt.ylabel(y_column)
                    else:
                        df.plot(kind='line', ax=plt.gca())
                        
                elif chart_type.lower() == 'bar':
                    if x_column and y_column:
                        plt.bar(df[x_column], df[y_column])
                        plt.xlabel(x_column)
                        plt.ylabel(y_column)
                        plt.xticks(rotation=45)
                    else:
                        df.plot(kind='bar', ax=plt.gca())
                        plt.xticks(rotation=45)
                        
                elif chart_type.lower() == 'scatter':
                    if x_column and y_column:
                        plt.scatter(df[x_column], df[y_column], alpha=0.6)
                        plt.xlabel(x_column)
                        plt.ylabel(y_column)
                    else:
                        # Use first two numeric columns
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        if len(numeric_cols) >= 2:
                            plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6)
                            plt.xlabel(numeric_cols[0])
                            plt.ylabel(numeric_cols[1])
                            
                elif chart_type.lower() == 'histogram':
                    if y_column:
                        plt.hist(df[y_column], bins=30, alpha=0.7, edgecolor='black')
                        plt.xlabel(y_column)
                        plt.ylabel('Frequency')
                    else:
                        # Use first numeric column
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        if len(numeric_cols) > 0:
                            plt.hist(df[numeric_cols[0]], bins=30, alpha=0.7, edgecolor='black')
                            plt.xlabel(numeric_cols[0])
                            plt.ylabel('Frequency')
                            
                elif chart_type.lower() == 'box':
                    if y_column:
                        plt.boxplot(df[y_column].dropna())
                        plt.ylabel(y_column)
                    else:
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        if len(numeric_cols) > 0:
                            plt.boxplot(df[numeric_cols[0]].dropna())
                            plt.ylabel(numeric_cols[0])
                            
                elif chart_type.lower() == 'heatmap':
                    # Create correlation heatmap for numeric columns
                    numeric_df = df.select_dtypes(include=['number'])
                    if not numeric_df.empty:
                        correlation_matrix = numeric_df.corr()
                        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
                        
                else:
                    # Default to line plot
                    df.plot(ax=plt.gca())
                
                # Set title and improve layout
                plt.title(title, fontsize=16, fontweight='bold', pad=20)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save the chart
                output_path = self.output_dir / f"{filename}.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close()  # Important: close the figure to free memory
                
                return f"Chart saved successfully as {filename}.png"
                
            except Exception as e:
                plt.close()  # Ensure figure is closed even on error
                return f"Error creating chart: {str(e)}"

# Configuration
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
CHARTS_DIR = BASE_DIR / "charts"
SESSIONS_DIR = BASE_DIR / "sessions"
INDEX_FILE = BASE_DIR / "static" / "index.html"

# Create directories
for dir_path in [UPLOAD_DIR, CHARTS_DIR, SESSIONS_DIR]:
    dir_path.mkdir(exist_ok=True)

# FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Data Analysis Chatbot...")
    yield
    print("Shutting down...")

app = FastAPI(title="Data Analysis Chatbot", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/charts", StaticFiles(directory=CHARTS_DIR), name="charts")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Serve the SPA
@app.get("/", response_class=HTMLResponse)
async def get_index():
    if INDEX_FILE.exists():
        return HTMLResponse(content=INDEX_FILE.read_text(encoding="utf-8"), status_code=200)
    raise HTTPException(status_code=404, detail="Index file not found")

# Models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    visualizations: List[str] = []

# Session Manager
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        if not session_id or session_id not in self.sessions:
            session_id = str(uuid.uuid4())
            memory_db = SqliteMemoryDb(
                table_name="user_memories",
                db_file=f"{SESSIONS_DIR}/{session_id}_memory.db"
            )
            memory = Memory(db=memory_db)
            storage = SqliteStorage(
                table_name="agent_sessions",
                db_file=f"{SESSIONS_DIR}/{session_id}_storage.db"
            )
            self.sessions[session_id] = {
                "memory": memory,
                "storage": storage,
                "csv_path": None,
                "data_schema": None,
                "chart_counter": 0
            }
        return session_id

    def get_session(self, session_id: str) -> Dict[str, Any]:
        return self.sessions.get(session_id, {})

    def update_csv(self, session_id: str, csv_path: str, schema: Dict):
        if session_id in self.sessions:
            self.sessions[session_id]["csv_path"] = csv_path
            self.sessions[session_id]["data_schema"] = schema

    def get_next_chart_name(self, session_id: str) -> str:
        if session_id in self.sessions:
            counter = self.sessions[session_id]["chart_counter"]
            self.sessions[session_id]["chart_counter"] = counter + 1
            return f"{session_id}_chart_{counter}"
        return f"{session_id}_chart_0"

# Initialize session manager
session_manager = SessionManager()

# Agent Factory
class AgentFactory:
    @staticmethod
    def create_schema_agent() -> Agent:
        """Agent for understanding data schema and structure"""
        return Agent(
            model=OpenAIChat(id="o3-mini"),
            tools=[PythonTools()],
            instructions=[
                "You are a data schema analyst.",
                "Analyze CSV files and understand their structure.",
                "Identify column types, missing values, and data patterns.",
                "Provide clear summaries of the data structure.",
                "Do not show code to the user, just results.",
            ],
            show_tool_calls=False,
        )

    @staticmethod
    def create_analytics_agent(memory: Memory, storage: SqliteStorage) -> Agent:
        """Agent for data analytics queries"""
        return Agent(
            model=OpenAIChat(id="o3-mini"),
            tools=[PythonTools()],
            memory=memory,
            storage=storage,
            add_history_to_messages=True,
            num_history_runs=3,
            enable_user_memories=True,
            instructions=[
                "You are a data analytics expert.",
                "Answer questions about data using pandas and statistical analysis.",
                "Provide insights and patterns in the data.",
                "Remember previous conversations and context.",
                "Do not show code execution details to the user.",
                "Format numerical results clearly.",
            ],
            show_tool_calls=False,
        )

    @staticmethod
    def create_visualization_agent(memory: Memory, storage: SqliteStorage, session_id: str) -> Agent:
        """Agent for creating data visualizations"""
        viz_tools = VisualizationTools(output_dir=str(CHARTS_DIR))
        return Agent(
            model=OpenAIChat(id="o3-mini"),
            tools=[viz_tools, PythonTools()],
            memory=memory,
            storage=storage,
            add_history_to_messages=True,
            num_history_runs=3,
            enable_user_memories=True,
            instructions=[
                "You are a data visualization expert.",
                "Create beautiful and informative charts based on user requests.",
                f"Always save charts with filenames starting with '{session_id}_chart_'",
                "Use the create_chart tool to generate visualizations.",
                "After creating a chart, simply confirm it was created without additional commentary.",
                "Focus on the chart creation, not lengthy explanations.",
                "Remember previous conversations and visualizations created.",
                "Do not show code or tool calls to the user.",
            ],
            show_tool_calls=False,
        )

    @staticmethod
    def create_router_agent() -> Agent:
        """Agent for routing queries to appropriate specialized agents"""
        return Agent(
            model=OpenAIChat(id="o3-mini"),
            instructions=[
                "You are a query router.",
                "Analyze user queries and determine the appropriate response type:",
                "1. 'schema' - for questions about data structure, columns, types",
                "2. 'analytics' - for statistical analysis, calculations, data insights",
                "3. 'visualization' - for creating charts, plots, or graphs",
                "4. 'general' - for general conversation or greetings",
                "Respond with only one word: schema, analytics, visualization, or general",
            ],
        )

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...), session_id: Optional[str] = None):
    """Upload a CSV file for analysis"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    # Get or create session
    session_id = session_manager.get_or_create_session(session_id)

    # Save file
    file_path = UPLOAD_DIR / f"{session_id}_{file.filename}"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Analyze schema
    try:
        df = pd.read_csv(file_path)
        schema = {
            "columns": list(df.columns),
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "sample": df.head(3).to_dict(orient='records')
        }

        # Update session
        session_manager.update_csv(session_id, str(file_path), schema)

        # Get initial analysis
        schema_agent = AgentFactory.create_schema_agent()
        analysis = schema_agent.run(
            f"Analyze this data schema and provide a summary: {json.dumps(schema, indent=2)}"
        )

        return JSONResponse({
            "session_id": session_id,
            "filename": file.filename,
            "schema": schema,
            "initial_analysis": analysis.content
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Chat endpoint for data analysis queries"""
    session_id = session_manager.get_or_create_session(message.session_id)
    session = session_manager.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get session data
    memory = session["memory"]
    storage = session["storage"]
    csv_path = session["csv_path"]
    data_schema = session["data_schema"]

    # Route the query
    router = AgentFactory.create_router_agent()
    query_type = router.run(message.message).content.strip().lower()

    visualizations = []

    try:
        if query_type == "visualization":
            # Create visualization agent with session context
            viz_agent = AgentFactory.create_visualization_agent(memory, storage, session_id)
            
            # Add context about the data
            context = ""
            if csv_path:
                context = f"\nThe user has uploaded a CSV file at: {csv_path}\n"
            if data_schema:
                context += f"Data schema: {json.dumps(data_schema['columns'])}\n"

            # Get current chart name
            chart_name = session_manager.get_next_chart_name(session_id)

            # Process request with specific filename instruction
            enhanced_message = f"{context}\n{message.message}\n\nIMPORTANT: Use filename '{chart_name}' when creating the chart."
            response = viz_agent.run(enhanced_message, user_id=session_id)

            # Check if chart was created
            chart_path = CHARTS_DIR / f"{chart_name}.png"
            if chart_path.exists():
                visualizations.append(f"/charts/{chart_name}.png")
                # Return a simple confirmation message instead of the full agent response
                return ChatResponse(
                    response="Chart created successfully! You can view it below.",
                    session_id=session_id,
                    visualizations=visualizations
                )
            else:
                return ChatResponse(
                    response=response.content,
                    session_id=session_id,
                    visualizations=[]
                )

        elif query_type == "analytics":
            # Create analytics agent
            analytics_agent = AgentFactory.create_analytics_agent(memory, storage)
            
            # Add context
            context = ""
            if csv_path:
                context = f"\nCSV file path: {csv_path}\n"
            if data_schema:
                context += f"Columns: {data_schema['columns']}\n"

            response = analytics_agent.run(f"{context}\n{message.message}", user_id=session_id)
            return ChatResponse(
                response=response.content,
                session_id=session_id,
                visualizations=[]
            )

        elif query_type == "schema":
            # Use schema agent
            schema_agent = AgentFactory.create_schema_agent()
            if data_schema:
                response = schema_agent.run(
                    f"Based on this schema: {json.dumps(data_schema, indent=2)}\n\n{message.message}"
                )
            else:
                response = schema_agent.run("No data has been uploaded yet. Please upload a CSV file first.")

            return ChatResponse(
                response=response.content if hasattr(response, 'content') else str(response),
                session_id=session_id,
                visualizations=[]
            )

        else:  # general
            # Simple response without specialized agent
            response_text = "Hello! I'm your data analysis assistant. Please upload a CSV file to get started, or ask me about data analysis in general."
            return ChatResponse(
                response=response_text,
                session_id=session_id,
                visualizations=[]
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
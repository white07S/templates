import os
from typing import Annotated, Literal, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Configure your vLLM endpoint
VLLM_BASE_URL = "http://localhost:8000/v1"  # Update this to your vLLM server URL
VLLM_API_KEY = "EMPTY"  # vLLM doesn't require API key, but the client needs something
MODEL_NAME = "your-model-name"  # Replace with your actual model name

# Initialize the LLM client pointing to vLLM
llm = ChatOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
    model=MODEL_NAME,
    temperature=0.7,
)

# Define simple tools for testing
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # Simulated weather data
    weather_data = {
        "new york": "Sunny, 72°F",
        "london": "Cloudy, 59°F", 
        "tokyo": "Rainy, 65°F",
        "paris": "Partly cloudy, 68°F"
    }
    return weather_data.get(location.lower(), f"Weather data not available for {location}")

@tool
def calculate(expression: str) -> str:
    """Perform basic mathematical calculations. 
    Input should be a valid Python mathematical expression."""
    try:
        # Use eval safely for basic math operations
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Simulated search results
    search_results = {
        "langchain": "LangChain is a framework for developing applications powered by language models.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "vllm": "vLLM is a fast and easy-to-use library for LLM inference and serving."
    }
    
    for key, value in search_results.items():
        if key in query.lower():
            return value
    
    return f"Search results for '{query}': No specific results found in our mock database."

# Import math for the calculator tool
import math

# Create the tools list
tools = [get_weather, calculate, search_web]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# Define the graph state
class AgentState(MessagesState):
    """State of the agent."""
    pass

# Define the function that calls the model
def call_model(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Define the function that determines whether to continue or not
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM makes a tool call, then we route to the "tools" node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return "__end__"

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

# Set the entrypoint
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "__end__": "__end__",
    }
)

# Add edge from tools back to agent
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile()

# Example usage
def run_agent(user_input: str):
    """Run the agent with a user input."""
    print(f"\nUser: {user_input}")
    
    # Create initial state with user message
    initial_state = {
        "messages": [HumanMessage(content=user_input)]
    }
    
    # Run the graph
    result = app.invoke(initial_state)
    
    # Print the final response
    final_message = result["messages"][-1].content
    print(f"Assistant: {final_message}")
    
    return result

# Test the agent with different queries
if __name__ == "__main__":
    print("=== LangGraph + vLLM Tool Calling Demo ===\n")
    
    # Test 1: Weather query
    run_agent("What's the weather like in New York?")
    
    # Test 2: Calculation
    run_agent("Calculate the square root of 144 plus 10")
    
    # Test 3: Web search
    run_agent("Search for information about LangGraph")
    
    # Test 4: Multiple tools in sequence
    run_agent("What's the weather in London and Tokyo? Also calculate 25 * 4")
    
    # Test 5: No tool needed
    run_agent("Hello! How are you doing today?")

    # Optional: Interactive mode
    print("\n=== Interactive Mode ===")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        run_agent(user_input)


pip install langchain langgraph langchain-openai

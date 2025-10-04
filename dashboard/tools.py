"""
Function calling tools for OpenAI integration.

This module provides a framework for defining and executing tools that can be called
by the OpenAI chat models. Tools are defined as functions with proper type hints
and docstrings, and automatically converted to OpenAI function schemas.

Example:
    @tool
    def get_weather(location: str, unit: str = "celsius") -> dict:
        '''Get current weather for a location.

        Args:
            location: The city and state/country
            unit: Temperature unit (celsius or fahrenheit)
        '''
        # Implementation here
        pass
"""

import json
import inspect
from typing import Dict, List, Any, Callable, Optional, get_type_hints
from functools import wraps
from datetime import datetime
import asyncio

from config import settings


class ToolRegistry:
    """Registry for managing OpenAI function calling tools."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, Dict] = {}

    def register(self, func: Callable) -> Callable:
        """Register a function as a tool."""
        self._tools[func.__name__] = func
        self._schemas[func.__name__] = self._generate_schema(func)
        return func

    def _generate_schema(self, func: Callable) -> Dict[str, Any]:
        """Generate OpenAI function schema from function signature."""
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Parse docstring
        doc = inspect.getdoc(func) or ""
        lines = doc.strip().split('\n')
        description = lines[0] if lines else func.__name__

        # Extract parameter descriptions from docstring
        param_descriptions = {}
        in_args_section = False
        for line in lines:
            line = line.strip()
            if line.lower().startswith('args:'):
                in_args_section = True
                continue
            if in_args_section and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    param_name = parts[0].strip()
                    param_desc = parts[1].strip()
                    param_descriptions[param_name] = param_desc

        # Build schema
        schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }

        # Add parameters
        for param_name, param in signature.parameters.items():
            param_type = type_hints.get(param_name, str)
            param_schema = self._python_type_to_json_schema(param_type)

            if param_name in param_descriptions:
                param_schema["description"] = param_descriptions[param_name]

            schema["function"]["parameters"]["properties"][param_name] = param_schema

            # Add to required if no default value
            if param.default == inspect.Parameter.empty:
                schema["function"]["parameters"]["required"].append(param_name)

        return schema

    def _python_type_to_json_schema(self, python_type) -> Dict[str, Any]:
        """Convert Python type to JSON schema type."""
        if python_type == str:
            return {"type": "string"}
        elif python_type == int:
            return {"type": "integer"}
        elif python_type == float:
            return {"type": "number"}
        elif python_type == bool:
            return {"type": "boolean"}
        elif python_type == list or str(python_type).startswith('typing.List'):
            return {"type": "array"}
        elif python_type == dict or str(python_type).startswith('typing.Dict'):
            return {"type": "object"}
        else:
            return {"type": "string"}  # Default fallback

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get all tool schemas for OpenAI API."""
        return list(self._schemas.values())

    def get_tool_names(self) -> List[str]:
        """Get list of all tool names."""
        return list(self._tools.keys())

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool with given arguments."""
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found")

        func = self._tools[tool_name]

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(**arguments)
            else:
                result = func(**arguments)
            return result
        except Exception as e:
            return {"error": str(e), "tool": tool_name, "arguments": arguments}


# Global tool registry
registry = ToolRegistry()


def tool(func: Callable) -> Callable:
    """Decorator to register a function as an OpenAI tool."""
    return registry.register(func)


# ============================================================================
# Built-in Tools
# ============================================================================

@tool
def get_current_time(timezone: str = "UTC") -> dict:
    """Get the current date and time.

    Args:
        timezone: Timezone to get time for (e.g., UTC, EST, PST)
    """
    # For simplicity, just return UTC time
    # In production, you'd handle actual timezone conversion
    current_time = datetime.utcnow()

    return {
        "current_time": current_time.isoformat(),
        "timezone": timezone,
        "formatted": current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "unix_timestamp": int(current_time.timestamp())
    }


@tool
def calculate_math(expression: str) -> dict:
    """Calculate a mathematical expression safely.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2", "sqrt(16)")
    """
    import math

    # Define safe functions
    safe_functions = {
        'abs', 'round', 'min', 'max', 'sum',
        'sqrt', 'pow', 'exp', 'log', 'log10',
        'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
        'degrees', 'radians', 'pi', 'e'
    }

    # Create safe namespace
    safe_dict = {"__builtins__": {}}
    safe_dict.update({name: getattr(math, name) for name in safe_functions if hasattr(math, name)})

    try:
        # Evaluate expression safely
        result = eval(expression, safe_dict, {})
        return {
            "expression": expression,
            "result": result,
            "type": type(result).__name__
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": str(e),
            "result": None
        }


@tool
async def search_memories(query: str, user_id: str, limit: int = 5) -> dict:
    """Search user memories for relevant information.

    Args:
        query: Search query to find relevant memories
        user_id: User ID to search memories for
        limit: Maximum number of memories to return
    """
    from memory_manager import MemoryOrchestrator

    try:
        orchestrator = MemoryOrchestrator()
        user_memory = orchestrator.get_user_memory(user_id)

        memories = await user_memory.search_memories(query, limit=limit)

        return {
            "query": query,
            "user_id": user_id,
            "found_memories": len(memories),
            "memories": [
                {
                    "id": mem.memory_id,
                    "content": mem.content,
                    "type": mem.memory_type,
                    "importance": mem.importance_score,
                    "created": mem.created_at.isoformat(),
                    "keywords": mem.keywords
                }
                for mem in memories
            ]
        }
    except Exception as e:
        return {
            "query": query,
            "user_id": user_id,
            "error": str(e),
            "memories": []
        }


@tool
async def get_memory_stats(user_id: str) -> dict:
    """Get statistics about a user's memories.

    Args:
        user_id: User ID to get memory statistics for
    """
    from memory_manager import MemoryOrchestrator

    try:
        orchestrator = MemoryOrchestrator()
        user_memory = orchestrator.get_user_memory(user_id)

        all_memories = await user_memory.memory_store.get_user_memories(user_id, limit=1000)

        # Calculate statistics
        memory_types = {}
        total_importance = 0
        recent_count = 0
        week_ago = datetime.utcnow().timestamp() - (7 * 24 * 3600)

        for memory in all_memories:
            # Count by type
            mem_type = memory.memory_type
            memory_types[mem_type] = memory_types.get(mem_type, 0) + 1

            # Sum importance
            total_importance += memory.importance_score

            # Count recent
            if memory.created_at.timestamp() > week_ago:
                recent_count += 1

        return {
            "user_id": user_id,
            "total_memories": len(all_memories),
            "memory_types": memory_types,
            "average_importance": total_importance / len(all_memories) if all_memories else 0,
            "recent_memories_7_days": recent_count
        }
    except Exception as e:
        return {
            "user_id": user_id,
            "error": str(e),
            "total_memories": 0
        }


@tool
def create_reminder(message: str, remind_at: str) -> dict:
    """Create a reminder for the user.

    Args:
        message: Reminder message
        remind_at: When to remind (ISO format datetime or relative like "in 1 hour")
    """
    # This is a placeholder implementation
    # In production, you'd integrate with a task scheduler

    return {
        "reminder_id": f"reminder_{datetime.utcnow().timestamp()}",
        "message": message,
        "remind_at": remind_at,
        "status": "created",
        "note": "Reminder system not fully implemented - this is a placeholder"
    }


@tool
def format_text(text: str, format_type: str = "markdown") -> dict:
    """Format text in various ways.

    Args:
        text: Text to format
        format_type: Format type (markdown, html, uppercase, lowercase, title)
    """
    try:
        if format_type == "markdown":
            # Simple markdown formatting
            formatted = f"**{text}**" if not text.startswith("**") else text
        elif format_type == "html":
            formatted = f"<p>{text}</p>"
        elif format_type == "uppercase":
            formatted = text.upper()
        elif format_type == "lowercase":
            formatted = text.lower()
        elif format_type == "title":
            formatted = text.title()
        else:
            formatted = text

        return {
            "original": text,
            "formatted": formatted,
            "format_type": format_type
        }
    except Exception as e:
        return {
            "original": text,
            "error": str(e),
            "format_type": format_type
        }


@tool
async def save_user_preference(user_id: str, preference_key: str, preference_value: str) -> dict:
    """Save a user preference to their memory.

    Args:
        user_id: User ID to save preference for
        preference_key: Key for the preference (e.g., "favorite_color", "timezone")
        preference_value: Value of the preference
    """
    from memory_manager import MemoryOrchestrator
    from models import MemoryType

    try:
        orchestrator = MemoryOrchestrator()
        user_memory = orchestrator.get_user_memory(user_id)

        # Create a preference memory
        content = f"User prefers {preference_key}: {preference_value}"

        memory = await user_memory.add_memory(
            content=content,
            memory_type=MemoryType.PROCEDURAL,
            importance_score=0.7,
            keywords=[preference_key, "preference", "setting"]
        )

        return {
            "user_id": user_id,
            "preference_key": preference_key,
            "preference_value": preference_value,
            "memory_id": memory.memory_id,
            "status": "saved"
        }
    except Exception as e:
        return {
            "user_id": user_id,
            "preference_key": preference_key,
            "preference_value": preference_value,
            "error": str(e),
            "status": "failed"
        }


@tool
def generate_random_fact(category: str = "general") -> dict:
    """Generate a random fact from a specific category.

    Args:
        category: Category of fact (science, history, nature, technology, general)
    """
    import random

    facts_db = {
        "science": [
            "A single cloud can weigh more than a million pounds.",
            "Octopuses have three hearts and blue blood.",
            "There are more possible games of chess than atoms in the observable universe.",
            "Honey never spoils - archaeologists have found edible honey in ancient tombs.",
            "A group of flamingos is called a 'flamboyance'."
        ],
        "history": [
            "Cleopatra lived closer in time to the Moon landing than to the construction of the Great Pyramid.",
            "The Great Wall of China isn't visible from space with the naked eye.",
            "Vikings never actually wore horned helmets.",
            "Napoleon was actually average height for his time period.",
            "The first known vending machine was invented in ancient Greece and dispensed holy water."
        ],
        "nature": [
            "Bananas are berries, but strawberries aren't.",
            "A group of owls is called a 'parliament'.",
            "Sharks have been around longer than trees.",
            "Wombat droppings are cube-shaped.",
            "A day on Venus is longer than its year."
        ],
        "technology": [
            "The first computer bug was an actual bug - a moth stuck in a computer relay.",
            "More people have mobile phones than have access to clean water.",
            "The first webcam was created to monitor a coffee pot.",
            "Email existed before the World Wide Web.",
            "The @ symbol is over 500 years old."
        ]
    }

    # Get facts for category or all facts if category not found
    category_facts = facts_db.get(category, [])
    if not category_facts:
        # If category not found, pick from all facts
        all_facts = []
        for cat_facts in facts_db.values():
            all_facts.extend(cat_facts)
        category_facts = all_facts
        used_category = "general"
    else:
        used_category = category

    if category_facts:
        fact = random.choice(category_facts)
    else:
        fact = "Did you know that asking for random facts is a great way to learn something new?"

    return {
        "fact": fact,
        "category": used_category,
        "available_categories": list(facts_db.keys())
    }


@tool
async def create_task_list(tasks: str, user_id: str) -> dict:
    """Create and save a task list for the user.

    Args:
        tasks: Comma-separated list of tasks or task description
        user_id: User ID to save tasks for
    """
    from memory_manager import MemoryOrchestrator
    from models import MemoryType

    try:
        # Parse tasks
        if "," in tasks:
            task_list = [task.strip() for task in tasks.split(",")]
        else:
            # Single task or description
            task_list = [tasks.strip()]

        # Save as procedural memory
        orchestrator = MemoryOrchestrator()
        user_memory = orchestrator.get_user_memory(user_id)

        content = f"Task list created: {', '.join(task_list)}"
        memory = await user_memory.add_memory(
            content=content,
            memory_type=MemoryType.PROCEDURAL,
            importance_score=0.8,
            keywords=["tasks", "todo", "planning"] + task_list[:3]
        )

        return {
            "tasks": task_list,
            "task_count": len(task_list),
            "memory_id": memory.memory_id,
            "status": "created"
        }
    except Exception as e:
        return {
            "tasks": tasks,
            "error": str(e),
            "status": "failed"
        }


@tool
def generate_plan(goal: str, timeframe: str = "1 week") -> dict:
    """Generate a simple plan to achieve a goal.

    Args:
        goal: The goal to achieve
        timeframe: Timeframe for the goal (e.g., "1 week", "1 month")
    """
    try:
        # Simple plan generation logic
        steps = []

        if "learn" in goal.lower():
            steps = [
                "Research the topic and gather learning resources",
                "Create a study schedule",
                "Start with basics and fundamentals",
                "Practice regularly",
                "Review and assess progress"
            ]
        elif "project" in goal.lower() or "build" in goal.lower():
            steps = [
                "Define project requirements and scope",
                "Plan the architecture and design",
                "Break down into smaller tasks",
                "Start implementation",
                "Test and iterate",
                "Finalize and deploy"
            ]
        elif "habit" in goal.lower() or "routine" in goal.lower():
            steps = [
                "Start small with minimal viable habit",
                "Set up environmental triggers",
                "Track progress daily",
                "Gradually increase difficulty",
                "Establish accountability"
            ]
        else:
            # Generic plan
            steps = [
                "Define specific, measurable objectives",
                "Research and gather necessary resources",
                "Create a timeline with milestones",
                "Take the first step",
                "Monitor progress and adjust as needed"
            ]

        return {
            "goal": goal,
            "timeframe": timeframe,
            "steps": steps,
            "estimated_duration": timeframe
        }
    except Exception as e:
        return {
            "goal": goal,
            "error": str(e),
            "steps": []
        }


@tool
def analyze_text(text: str, analysis_type: str = "summary") -> dict:
    """Analyze text for various properties.

    Args:
        text: Text to analyze
        analysis_type: Type of analysis (summary, word_count, sentiment, keywords)
    """
    try:
        results = {"text_length": len(text), "word_count": len(text.split())}

        if analysis_type == "word_count":
            results.update({
                "characters": len(text),
                "characters_no_spaces": len(text.replace(" ", "")),
                "sentences": len([s for s in text.split(".") if s.strip()]),
                "paragraphs": len([p for p in text.split("\n\n") if p.strip()])
            })

        elif analysis_type == "keywords":
            # Simple keyword extraction
            words = text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Filter short words
                    word_freq[word] = word_freq.get(word, 0) + 1

            # Top 10 most frequent words
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            results["keywords"] = [word for word, freq in keywords]

        elif analysis_type == "sentiment":
            # Very basic sentiment analysis
            positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "like", "happy", "joy"]
            negative_words = ["bad", "terrible", "awful", "hate", "dislike", "sad", "angry", "frustrated", "disappointed"]

            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)

            if positive_count > negative_count:
                sentiment = "positive"
            elif negative_count > positive_count:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            results["sentiment"] = sentiment
            results["positive_indicators"] = positive_count
            results["negative_indicators"] = negative_count

        elif analysis_type == "summary":
            # Simple summary (first and last sentences)
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            if len(sentences) > 2:
                summary = f"{sentences[0]}. ... {sentences[-1]}."
            else:
                summary = text
            results["summary"] = summary

        return results

    except Exception as e:
        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "error": str(e),
            "analysis_type": analysis_type
        }


@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> dict:
    """Convert between different units.

    Args:
        value: Numeric value to convert
        from_unit: Source unit (e.g., "celsius", "fahrenheit", "meters", "feet")
        to_unit: Target unit
    """
    try:
        conversions = {
            # Temperature
            ("celsius", "fahrenheit"): lambda x: (x * 9/5) + 32,
            ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
            ("celsius", "kelvin"): lambda x: x + 273.15,
            ("kelvin", "celsius"): lambda x: x - 273.15,

            # Length
            ("meters", "feet"): lambda x: x * 3.28084,
            ("feet", "meters"): lambda x: x / 3.28084,
            ("meters", "inches"): lambda x: x * 39.3701,
            ("inches", "meters"): lambda x: x / 39.3701,
            ("kilometers", "miles"): lambda x: x * 0.621371,
            ("miles", "kilometers"): lambda x: x / 0.621371,

            # Weight
            ("kilograms", "pounds"): lambda x: x * 2.20462,
            ("pounds", "kilograms"): lambda x: x / 2.20462,
            ("grams", "ounces"): lambda x: x * 0.035274,
            ("ounces", "grams"): lambda x: x / 0.035274,
        }

        key = (from_unit.lower(), to_unit.lower())
        if key in conversions:
            result = conversions[key](value)
            return {
                "original_value": value,
                "original_unit": from_unit,
                "converted_value": round(result, 4),
                "converted_unit": to_unit,
                "formula": f"{value} {from_unit} = {round(result, 4)} {to_unit}"
            }
        else:
            return {
                "original_value": value,
                "original_unit": from_unit,
                "target_unit": to_unit,
                "error": f"Conversion from {from_unit} to {to_unit} not supported",
                "available_conversions": list(set([f"{k[0]} -> {k[1]}" for k in conversions.keys()]))
            }

    except Exception as e:
        return {
            "original_value": value,
            "error": str(e),
            "from_unit": from_unit,
            "to_unit": to_unit
        }


# ============================================================================
# Tool Execution Helper
# ============================================================================

class ToolExecutor:
    """Helper class for executing tools in OpenAI function calling workflow."""

    def __init__(self, tool_registry: ToolRegistry = None):
        self.registry = tool_registry or registry

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get all available tools for OpenAI API."""
        return self.registry.get_tools()

    async def process_tool_calls(self, tool_calls: List[Dict[str, Any]], user_id: str = None) -> List[Dict[str, Any]]:
        """Process a list of tool calls from OpenAI and return results."""
        results = []

        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id")
            function_name = tool_call.get("function", {}).get("name")
            function_args = tool_call.get("function", {}).get("arguments", "{}")

            try:
                # Parse arguments
                if isinstance(function_args, str):
                    arguments = json.loads(function_args)
                else:
                    arguments = function_args

                # Inject user_id for tools that need it
                if user_id and function_name in ["search_memories", "get_memory_stats", "save_user_preference"]:
                    arguments["user_id"] = user_id

                # Execute tool
                result = await self.registry.execute_tool(function_name, arguments)

                # Format result
                if isinstance(result, dict):
                    content = json.dumps(result, indent=2)
                else:
                    content = str(result)

                results.append({
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "name": function_name,
                    "content": content
                })

            except Exception as e:
                # Return error as tool result
                error_content = json.dumps({
                    "error": str(e),
                    "function": function_name,
                    "arguments": function_args
                }, indent=2)

                results.append({
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "name": function_name,
                    "content": error_content
                })

        return results


# Global executor instance
executor = ToolExecutor()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example of how to use the tools

    # Print available tools
    print("Available tools:")
    for tool_schema in registry.get_tools():
        print(f"- {tool_schema['function']['name']}: {tool_schema['function']['description']}")

    # Example tool execution
    async def test_tools():
        # Test math calculation
        result = await registry.execute_tool("calculate_math", {"expression": "2 + 2 * 3"})
        print(f"Math result: {result}")

        # Test time
        result = await registry.execute_tool("get_current_time", {"timezone": "UTC"})
        print(f"Time result: {result}")

    # Run test
    import asyncio
    asyncio.run(test_tools())

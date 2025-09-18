import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone
from abc import ABC, abstractmethod
import aiofiles
import os
from pathlib import Path


class BaseTool(ABC):
    """Base class for all tools"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        pass

    @property
    def schema(self) -> Dict[str, Any]:
        """Return OpenAI function calling schema for this tool"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters_schema()
            }
        }

    @abstractmethod
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for the tool's parameters"""
        pass


class CalculatorTool(BaseTool):
    """Simple calculator tool for mathematical operations"""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations. Supports basic arithmetic operations."
        )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4')"
                }
            },
            "required": ["expression"]
        }

    async def execute(self, expression: str) -> Dict[str, Any]:
        """Execute mathematical calculation"""
        try:
            # Basic safety check - only allow mathematical operations
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return {
                    "error": "Invalid characters in expression. Only numbers and basic operators allowed."
                }

            # Evaluate the expression
            result = eval(expression)
            return {
                "result": result,
                "expression": expression,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {
                "error": f"Calculation error: {str(e)}",
                "expression": expression
            }


class FileSearchTool(BaseTool):
    """Tool to search for files in a directory"""

    def __init__(self, search_base_path: str = "./"):
        super().__init__(
            name="file_search",
            description="Search for files and directories by name pattern"
        )
        self.search_base_path = Path(search_base_path)

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "File name pattern to search for (supports wildcards)"
                },
                "path": {
                    "type": "string",
                    "description": "Optional: Directory path to search in (relative to base path)"
                },
                "file_type": {
                    "type": "string",
                    "enum": ["file", "directory", "both"],
                    "description": "Type of items to search for"
                }
            },
            "required": ["pattern"]
        }

    async def execute(self, pattern: str, path: str = ".", file_type: str = "both") -> Dict[str, Any]:
        """Search for files matching the pattern"""
        try:
            search_path = self.search_base_path / path
            if not search_path.exists():
                return {"error": f"Path does not exist: {path}"}

            results = []
            for item in search_path.rglob(pattern):
                item_type = "directory" if item.is_dir() else "file"

                if file_type == "both" or file_type == item_type:
                    results.append({
                        "name": item.name,
                        "path": str(item.relative_to(self.search_base_path)),
                        "type": item_type,
                        "size": item.stat().st_size if item.is_file() else None,
                        "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    })

            return {
                "results": results,
                "count": len(results),
                "pattern": pattern,
                "search_path": path
            }

        except Exception as e:
            return {"error": f"Search error: {str(e)}"}


class TimeWeatherTool(BaseTool):
    """Tool to get current time and mock weather information"""

    def __init__(self):
        super().__init__(
            name="time_weather",
            description="Get current time and weather information"
        )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Location for weather information (optional)"
                },
                "include_time": {
                    "type": "boolean",
                    "description": "Whether to include current time"
                },
                "include_weather": {
                    "type": "boolean",
                    "description": "Whether to include weather information"
                }
            },
            "required": []
        }

    async def execute(
        self,
        location: str = "Unknown",
        include_time: bool = True,
        include_weather: bool = False
    ) -> Dict[str, Any]:
        """Get time and weather information"""
        result = {}

        if include_time:
            result["current_time"] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "timezone": "UTC",
                "formatted": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            }

        if include_weather:
            # Mock weather data
            import random
            conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "clear"]
            result["weather"] = {
                "location": location,
                "condition": random.choice(conditions),
                "temperature": f"{random.randint(15, 30)}°C",
                "humidity": f"{random.randint(30, 80)}%",
                "note": "This is mock weather data for demonstration"
            }

        return result


class MemoryTool(BaseTool):
    """Tool to interact with the memory system"""

    def __init__(self, memory_manager):
        super().__init__(
            name="memory_tool",
            description="Add, retrieve, or search memories for the current user"
        )
        self.memory_manager = memory_manager

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "search", "get_stats"],
                    "description": "Action to perform with memory"
                },
                "content": {
                    "type": "string",
                    "description": "Content to add to memory (for 'add' action)"
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for 'search' action)"
                },
                "importance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Importance score for memory (0.0 to 1.0)"
                },
                "kind": {
                    "type": "string",
                    "enum": ["profile", "episodic"],
                    "description": "Type of memory to add"
                }
            },
            "required": ["action"]
        }

    async def execute(
        self,
        action: str,
        content: str = None,
        query: str = None,
        importance: float = 0.5,
        kind: str = "episodic",
        user_id: str = "default_user",
        session_id: str = None
    ) -> Dict[str, Any]:
        """Execute memory operation"""
        from uuid import UUID
        try:
            # Convert session_id to UUID if it's a string
            if session_id and isinstance(session_id, str):
                try:
                    session_id = UUID(session_id)
                except:
                    pass  # Keep as string if not a valid UUID

            if action == "add":
                if not content:
                    return {"error": "Content is required for 'add' action"}

                memory_id = await self.memory_manager.add_memory(
                    text=content,
                    user_id=user_id,
                    session_id=session_id,
                    kind=kind,
                    importance=importance
                )

                if memory_id:
                    return {
                        "success": True,
                        "memory_id": memory_id,
                        "action": "add",
                        "content": content
                    }
                else:
                    return {"error": "Failed to add memory (may not pass quality gates)"}

            elif action == "search":
                if not query:
                    return {"error": "Query is required for 'search' action"}

                memories = await self.memory_manager.retrieve_memories(
                    query=query,
                    user_id=user_id,
                    k=5
                )

                return {
                    "results": memories,
                    "count": len(memories),
                    "query": query
                }

            elif action == "get_stats":
                stats = await self.memory_manager.get_user_memory_stats(user_id)
                return {"stats": stats, "user_id": user_id}

            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            return {"error": f"Memory operation failed: {str(e)}"}


class NoteTakingTool(BaseTool):
    """Tool to create and manage notes"""

    def __init__(self, notes_dir: str = "./data/notes"):
        super().__init__(
            name="note_taking",
            description="Create, read, update, and list notes"
        )
        self.notes_dir = Path(notes_dir)
        self.notes_dir.mkdir(parents=True, exist_ok=True)

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "read", "update", "list", "delete"],
                    "description": "Action to perform with notes"
                },
                "title": {
                    "type": "string",
                    "description": "Note title"
                },
                "content": {
                    "type": "string",
                    "description": "Note content"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for the note"
                }
            },
            "required": ["action"]
        }

    async def execute(
        self,
        action: str,
        title: str = None,
        content: str = None,
        tags: List[str] = None
    ) -> Dict[str, Any]:
        """Execute note operation"""
        try:
            if action == "create":
                if not title or not content:
                    return {"error": "Title and content are required for creating a note"}

                note_file = self.notes_dir / f"{title.replace(' ', '_')}.json"
                note_data = {
                    "title": title,
                    "content": content,
                    "tags": tags or [],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }

                async with aiofiles.open(note_file, 'w') as f:
                    await f.write(json.dumps(note_data, indent=2))

                return {
                    "success": True,
                    "action": "create",
                    "title": title,
                    "file": str(note_file.name)
                }

            elif action == "read":
                if not title:
                    return {"error": "Title is required for reading a note"}

                note_file = self.notes_dir / f"{title.replace(' ', '_')}.json"
                if not note_file.exists():
                    return {"error": f"Note '{title}' not found"}

                async with aiofiles.open(note_file, 'r') as f:
                    note_data = json.loads(await f.read())

                return {"note": note_data}

            elif action == "list":
                notes = []
                for note_file in self.notes_dir.glob("*.json"):
                    async with aiofiles.open(note_file, 'r') as f:
                        note_data = json.loads(await f.read())
                        notes.append({
                            "title": note_data["title"],
                            "created_at": note_data["created_at"],
                            "tags": note_data.get("tags", [])
                        })

                return {"notes": notes, "count": len(notes)}

            elif action == "update":
                if not title:
                    return {"error": "Title is required for updating a note"}

                note_file = self.notes_dir / f"{title.replace(' ', '_')}.json"
                if not note_file.exists():
                    return {"error": f"Note '{title}' not found"}

                async with aiofiles.open(note_file, 'r') as f:
                    note_data = json.loads(await f.read())

                if content:
                    note_data["content"] = content
                if tags is not None:
                    note_data["tags"] = tags
                note_data["updated_at"] = datetime.now(timezone.utc).isoformat()

                async with aiofiles.open(note_file, 'w') as f:
                    await f.write(json.dumps(note_data, indent=2))

                return {"success": True, "action": "update", "title": title}

            elif action == "delete":
                if not title:
                    return {"error": "Title is required for deleting a note"}

                note_file = self.notes_dir / f"{title.replace(' ', '_')}.json"
                if not note_file.exists():
                    return {"error": f"Note '{title}' not found"}

                note_file.unlink()
                return {"success": True, "action": "delete", "title": title}

            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            return {"error": f"Note operation failed: {str(e)}"}


class ConversationHistoryTool(BaseTool):
    """Tool to retrieve conversation history"""

    def __init__(self, session_manager):
        super().__init__(
            name="conversation_history",
            description="Retrieve previous messages and conversation history from the current or past sessions"
        )
        self.session_manager = session_manager

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["last_message", "recent_messages", "search", "session_list"],
                    "description": "Type of history query"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Number of messages to retrieve"
                },
                "search_term": {
                    "type": "string",
                    "description": "Search term to find in messages (for 'search' query_type)"
                }
            },
            "required": ["query_type"]
        }

    async def execute(
        self,
        query_type: str,
        limit: int = 10,
        search_term: str = None,
        user_id: str = "default_user",
        session_id: str = None
    ) -> Dict[str, Any]:
        """Retrieve conversation history"""
        from uuid import UUID
        try:
            # Convert session_id to UUID if it's a string
            if session_id and isinstance(session_id, str):
                try:
                    session_id = UUID(session_id)
                except:
                    pass  # Keep as string if not a valid UUID

            if query_type == "last_message":
                # Get the last user message
                messages = await self.session_manager.get_session_messages(
                    user_id=user_id,
                    session_id=session_id,
                    limit=2  # Get last exchange
                )

                # Find the last user message
                for msg in reversed(messages):
                    if msg.role.value == "user":
                        return {
                            "last_user_message": msg.content,
                            "timestamp": msg.timestamp.isoformat(),
                            "found": True
                        }

                return {"found": False, "message": "No previous user message found"}

            elif query_type == "recent_messages":
                messages = await self.session_manager.get_session_messages(
                    user_id=user_id,
                    session_id=session_id,
                    limit=limit
                )

                return {
                    "messages": [
                        {
                            "role": msg.role.value,
                            "content": msg.content,
                            "timestamp": msg.timestamp.isoformat()
                        }
                        for msg in messages
                    ],
                    "count": len(messages)
                }

            elif query_type == "search":
                if not search_term:
                    return {"error": "search_term is required for search query_type"}

                # Get all messages from the session
                messages = await self.session_manager.get_session_messages(
                    user_id=user_id,
                    session_id=session_id,
                    limit=100  # Get more messages for search
                )

                # Search for messages containing the search term
                matching_messages = []
                for msg in messages:
                    if search_term.lower() in msg.content.lower():
                        matching_messages.append({
                            "role": msg.role.value,
                            "content": msg.content,
                            "timestamp": msg.timestamp.isoformat()
                        })

                return {
                    "search_term": search_term,
                    "matches": matching_messages,
                    "count": len(matching_messages)
                }

            elif query_type == "session_list":
                sessions = await self.session_manager.list_user_sessions(user_id)
                return {
                    "sessions": [
                        {
                            "session_id": str(session.session_id),
                            "created_at": session.created_at.isoformat(),
                            "last_activity": session.last_activity.isoformat(),
                            "message_count": session.message_count
                        }
                        for session in sessions
                    ],
                    "count": len(sessions)
                }

            else:
                return {"error": f"Unknown query_type: {query_type}"}

        except Exception as e:
            return {"error": f"Failed to retrieve conversation history: {str(e)}"}


class ToolRegistry:
    """Registry for managing available tools"""

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool):
        """Register a tool"""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all available tool names"""
        return list(self.tools.keys())

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get OpenAI function calling schema for all tools"""
        return [tool.schema for tool in self.tools.values()]

    async def execute_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name"""
        tool = self.get_tool(name)
        if not tool:
            return {"error": f"Tool '{name}' not found"}

        return await tool.execute(**kwargs)
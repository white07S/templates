#!/usr/bin/env python3
"""
CLI Testing Application for Chat Backend with Dual-Level Memory

This interactive CLI allows you to test the chat backend system with session
and user-level memory using GPT-4o and text-embedding-3-large.

Features:
- Interactive chat sessions
- Session management
- Memory inspection
- Performance monitoring
- Multiple users and sessions support
"""

import asyncio
import uuid
import json
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from datetime import datetime
from typing import Optional, Dict, Any
import requests
import sys

# Rich console for beautiful output
console = Console()

class ChatClient:
    """CLI client for interacting with the chat backend."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.current_session_id = None
        self.current_user_id = None
        self.session_history = []

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request to the backend."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.request(method, url, **kwargs)
            return response
        except requests.exceptions.ConnectionError:
            console.print("[red]Error: Cannot connect to backend. Is the server running?[/red]")
            console.print(f"[yellow]Trying to connect to: {url}[/yellow]")
            sys.exit(1)

    def start_new_session(self, user_id: Optional[str] = None):
        """Start a new chat session."""
        if user_id is None:
            user_id = Prompt.ask("Enter user ID", default="test_user")

        self.current_user_id = user_id
        self.current_session_id = str(uuid.uuid4())
        self.session_history = []

        console.print(Panel(
            f"Started new session:\n"
            f"User ID: [cyan]{self.current_user_id}[/cyan]\n"
            f"Session ID: [cyan]{self.current_session_id}[/cyan]",
            title="New Session",
            border_style="green"
        ))

    def send_message(self, message: str) -> Dict[str, Any]:
        """Send a message to the chat endpoint."""
        if not self.current_session_id or not self.current_user_id:
            console.print("[red]No active session. Start a new session first.[/red]")
            return {}

        payload = {
            "session_id": self.current_session_id,
            "user_id": self.current_user_id,
            "message": message
        }

        response = self._make_request("POST", "/chat", json=payload)

        if response.status_code == 200:
            data = response.json()
            self.session_history.append({
                "user": message,
                "assistant": data["response"],
                "timestamp": datetime.now().isoformat(),
                "metadata": data.get("metadata", {})
            })
            return data
        else:
            console.print(f"[red]Error {response.status_code}: {response.text}[/red]")
            return {}

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics for the current user."""
        if not self.current_user_id:
            console.print("[red]No active user.[/red]")
            return {}

        response = self._make_request("GET", f"/memory/stats/{self.current_user_id}")

        if response.status_code == 200:
            return response.json()
        else:
            console.print(f"[red]Error {response.status_code}: {response.text}[/red]")
            return {}

    def get_session_history(self) -> Dict[str, Any]:
        """Get session history from the backend."""
        if not self.current_session_id or not self.current_user_id:
            console.print("[red]No active session.[/red]")
            return {}

        response = self._make_request(
            "GET",
            f"/session/history/{self.current_session_id}",
            params={"user_id": self.current_user_id}
        )

        if response.status_code == 200:
            return response.json()
        else:
            console.print(f"[red]Error {response.status_code}: {response.text}[/red]")
            return {}

    def health_check(self) -> bool:
        """Check if the backend is healthy."""
        try:
            response = self._make_request("GET", "/health")
            return response.status_code == 200
        except:
            return False

def display_welcome():
    """Display welcome message."""
    welcome_text = """
# Chat Backend with Dual-Level Memory - CLI Test Tool

This CLI tool allows you to test the advanced chat backend system featuring:

- **Session Memory**: Conversation context within a single session
- **User Memory**: Persistent knowledge across sessions
- **Hybrid Search**: Combining lexical (Tantivy) and semantic (FAISS) search
- **OpenAI Integration**: GPT-4o for chat, text-embedding-3-large for embeddings

## Commands Available:
- `chat` - Start interactive chat session
- `stats` - View memory statistics
- `history` - View session history
- `new-session` - Start new session
- `tools` - List available function calling tools
- `react` - Test ReAct agent with complex queries
- `health` - Check backend status
- `help` - Show this help
- `exit` - Exit the application
    """
    console.print(Markdown(welcome_text))

def display_response(response_data: Dict[str, Any]):
    """Display chat response with metadata."""
    if not response_data:
        return

    # Main response
    console.print(Panel(
        response_data.get("response", "No response"),
        title="[bold blue]Assistant Response[/bold blue]",
        border_style="blue"
    ))

    # Metadata table
    metadata = response_data.get("metadata", {})
    if metadata:
        table = Table(title="Response Metadata", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Processing Time", f"{metadata.get('processing_time_ms', 0):.1f} ms")
        table.add_row("Context Tokens", str(metadata.get('context_tokens', 0)))
        table.add_row("Memory Tokens", str(metadata.get('memory_tokens', 0)))
        table.add_row("Memories Used", str(metadata.get('memories_count', 0)))
        table.add_row("Model", metadata.get('model', 'Unknown'))

        # Add ReAct information if available
        if metadata.get('used_react_agent'):
            table.add_row("Agent Type", metadata.get('agent_type', 'react'))
            table.add_row("Task Type", metadata.get('task_type', 'unknown'))
            table.add_row("Execution Mode", metadata.get('execution_mode', 'unknown'))
            table.add_row("Actions Executed", str(metadata.get('actions_executed', 0)))
            table.add_row("Actions Successful", str(metadata.get('actions_successful', 0)))

        console.print(table)

    # Show memories used
    memories_used = response_data.get("memories_used", [])
    if memories_used:
        console.print(f"[dim]Using {len(memories_used)} memories: {', '.join(memories_used[:3])}{'...' if len(memories_used) > 3 else ''}[/dim]")

def display_memory_stats(stats: Dict[str, Any]):
    """Display memory statistics."""
    if not stats:
        return

    console.print(Panel(
        f"User: [cyan]{stats.get('user_id', 'Unknown')}[/cyan]\n"
        f"Total Memories: [yellow]{stats.get('total_memories', 0)}[/yellow]\n"
        f"Average Importance: [yellow]{stats.get('avg_importance', 0):.2f}[/yellow]\n"
        f"Recent Memories (7 days): [yellow]{stats.get('recent_memories', 0)}[/yellow]",
        title="Memory Statistics",
        border_style="green"
    ))

    # Memory types breakdown
    memory_types = stats.get('memory_types', {})
    if memory_types:
        table = Table(title="Memory Types", show_header=True)
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="yellow")

        for mem_type, count in memory_types.items():
            table.add_row(mem_type.title(), str(count))

        console.print(table)

def display_session_history(history: Dict[str, Any]):
    """Display session conversation history."""
    messages = history.get('messages', [])

    if not messages:
        console.print("[yellow]No messages in current session.[/yellow]")
        return

    console.print(Panel(
        f"Session: [cyan]{history.get('session_id', 'Unknown')}[/cyan]\n"
        f"Messages: [yellow]{len(messages)}[/yellow]\n"
        f"Created: [yellow]{history.get('created_at', 'Unknown')}[/yellow]",
        title="Session Info",
        border_style="blue"
    ))

    # Show last few messages
    recent_messages = messages[-5:]  # Last 5 messages
    for msg in recent_messages:
        role = msg['role']
        content = msg['content']
        timestamp = msg['timestamp']

        style = "green" if role == "user" else "blue"
        console.print(f"[{style}]{role.upper()}[/{style}] ({timestamp}): {content[:100]}{'...' if len(content) > 100 else ''}")

@click.command()
@click.option('--host', default='localhost', help='Backend host')
@click.option('--port', default=8000, help='Backend port')
@click.option('--user-id', help='Default user ID')
def main(host: str, port: int, user_id: Optional[str]):
    """Interactive CLI for testing the chat backend with dual-level memory."""
    base_url = f"http://{host}:{port}"
    client = ChatClient(base_url)

    # Check backend health
    console.print(f"[yellow]Connecting to backend at {base_url}...[/yellow]")
    if not client.health_check():
        console.print("[red]Backend health check failed! Make sure the server is running.[/red]")
        sys.exit(1)

    console.print("[green]✓ Backend is healthy![/green]")

    # Display welcome
    display_welcome()

    # Start with a session if user_id provided
    if user_id:
        client.start_new_session(user_id)

    while True:
        try:
            # Get command
            if client.current_session_id:
                prompt_text = f"[{client.current_user_id}:{client.current_session_id[:8]}] > "
            else:
                prompt_text = "> "

            command = Prompt.ask(prompt_text).strip()

            if not command:
                continue

            # Handle commands
            if command.lower() in ['exit', 'quit', 'q']:
                console.print("[yellow]Goodbye![/yellow]")
                break

            elif command.lower() in ['help', 'h']:
                display_welcome()

            elif command.lower() in ['health']:
                if client.health_check():
                    console.print("[green]✓ Backend is healthy[/green]")
                else:
                    console.print("[red]✗ Backend is not responding[/red]")

            elif command.lower() in ['tools']:
                # Show available tools
                response = client._make_request("GET", "/tools")
                if response.status_code == 200:
                    tools_data = response.json()
                    console.print(Panel(
                        f"Available Tools: [cyan]{tools_data['count']}[/cyan]\n" +
                        "\n".join([f"• {name}" for name in tools_data['tool_names']]),
                        title="Function Calling Tools",
                        border_style="blue"
                    ))
                else:
                    console.print("[red]Failed to fetch tools[/red]")

            elif command.lower() in ['react']:
                # Test ReAct agent directly
                test_query = Prompt.ask("Enter complex query to test ReAct agent")
                if test_query.strip():
                    console.print("[yellow]Testing ReAct agent...[/yellow]")
                    response = client._make_request("POST", "/debug/react",
                                                    params={"query": test_query, "user_id": client.current_user_id or "test_user"})
                    if response.status_code == 200:
                        data = response.json()
                        console.print(Panel(
                            data.get('response', 'No response'),
                            title="ReAct Agent Response",
                            border_style="green"
                        ))

                        # Show metadata
                        metadata = data.get('metadata', {})
                        if metadata:
                            table = Table(title="ReAct Execution Details")
                            table.add_column("Property", style="cyan")
                            table.add_column("Value", style="yellow")

                            for key, value in metadata.items():
                                table.add_row(key.replace('_', ' ').title(), str(value))

                            console.print(table)
                    else:
                        console.print(f"[red]ReAct test failed: {response.text}[/red]")

            elif command.lower() in ['new-session', 'new', 'ns']:
                client.start_new_session()

            elif command.lower() in ['stats', 'memory', 'm']:
                stats = client.get_memory_stats()
                display_memory_stats(stats)

            elif command.lower() in ['history', 'hist', 'h']:
                history = client.get_session_history()
                display_session_history(history)

            elif command.lower() in ['chat', 'c']:
                if not client.current_session_id:
                    console.print("[yellow]Starting new session for chat...[/yellow]")
                    client.start_new_session()

                console.print("[green]Entering chat mode. Type 'exit' to return to command mode.[/green]")

                while True:
                    user_input = Prompt.ask("[bold green]You[/bold green]")

                    if user_input.lower() in ['exit', 'quit']:
                        break

                    if not user_input.strip():
                        continue

                    # Send message and display response
                    console.print("[yellow]Thinking...[/yellow]")
                    response = client.send_message(user_input)
                    display_response(response)

            else:
                # Treat as direct chat message
                if not client.current_session_id:
                    console.print("[yellow]Starting new session...[/yellow]")
                    client.start_new_session()

                console.print("[yellow]Thinking...[/yellow]")
                response = client.send_message(command)
                display_response(response)

        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'exit' to quit.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    main()
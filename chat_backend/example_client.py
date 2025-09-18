#!/usr/bin/env python3
"""
Example client for testing the chat backend API
"""

import asyncio
import json
import uuid
from typing import AsyncGenerator
import httpx


class ChatClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = str(uuid.uuid4())
        self.user_id = "test_user"

    async def health_check(self) -> dict:
        """Check API health"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/health")
            return response.json()

    async def list_tools(self) -> dict:
        """List available tools"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/tools")
            return response.json()

    async def send_message(
        self,
        message: str,
        reasoning_mode: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> AsyncGenerator[dict, None]:
        """Send a message and stream the response"""
        payload = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "message": message,
            "reasoning_mode": reasoning_mode,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat",
                json=payload,
                headers={"Accept": "text/plain"}
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])  # Remove "data: " prefix
                            yield data
                        except json.JSONDecodeError:
                            continue

    async def get_session_messages(self, limit: int = 50) -> dict:
        """Get messages from current session"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/sessions/{self.user_id}/{self.session_id}",
                params={"limit": limit}
            )
            return response.json()

    async def get_memory_stats(self) -> dict:
        """Get memory statistics"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/memory/{self.user_id}/stats")
            return response.json()


async def test_basic_chat():
    """Test basic chat functionality"""
    print("🧪 Testing Basic Chat...")

    client = ChatClient()

    # Health check
    health = await client.health_check()
    print(f"Health Status: {health['status']}")

    # List tools
    tools = await client.list_tools()
    print(f"Available Tools: {[tool['function']['name'] for tool in tools['tools']]}")

    # Send a simple message
    print("\n💬 Sending: 'Hello! Can you tell me what tools you have available?'")
    async for chunk in client.send_message("Hello! Can you tell me what tools you have available?"):
        if chunk['type'] == 'content':
            print(chunk['content'], end='', flush=True)
        elif chunk['type'] == 'complete':
            print(f"\n✅ Message completed: {chunk['message_id']}")
        elif chunk['type'] == 'error':
            print(f"\n❌ Error: {chunk['error']}")

    print("\n" + "="*50)


async def test_calculator():
    """Test calculator tool"""
    print("🧮 Testing Calculator Tool...")

    client = ChatClient()

    print("\n💬 Sending: 'Can you calculate 15 * 23 + 45?'")
    async for chunk in client.send_message("Can you calculate 15 * 23 + 45?"):
        if chunk['type'] == 'content':
            print(chunk['content'], end='', flush=True)
        elif chunk['type'] == 'complete':
            print(f"\n✅ Calculation completed")

    print("\n" + "="*50)


async def test_memory():
    """Test memory functionality"""
    print("🧠 Testing Memory System...")

    client = ChatClient()

    # Add some information to memory
    print("\n💬 Sending: 'Please remember that I prefer concise answers and work as a software engineer.'")
    async for chunk in client.send_message("Please remember that I prefer concise answers and work as a software engineer."):
        if chunk['type'] == 'content':
            print(chunk['content'], end='', flush=True)
        elif chunk['type'] == 'complete':
            print(f"\n✅ Memory update completed")

    # Test memory recall
    print("\n💬 Sending: 'What do you know about my preferences?'")
    async for chunk in client.send_message("What do you know about my preferences?"):
        if chunk['type'] == 'content':
            print(chunk['content'], end='', flush=True)
        elif chunk['type'] == 'complete':
            print(f"\n✅ Memory recall completed")

    # Get memory stats
    stats = await client.get_memory_stats()
    print(f"\n📊 Memory Stats: {stats}")

    print("\n" + "="*50)


async def test_reasoning_mode():
    """Test reasoning mode"""
    print("🤔 Testing Reasoning Mode...")

    client = ChatClient()

    print("\n💬 Sending (Reasoning Mode): 'I need to plan a birthday party for 20 people. Help me figure out what I need and estimate costs.'")

    async for chunk in client.send_message(
        "I need to plan a birthday party for 20 people. Help me figure out what I need and estimate costs.",
        reasoning_mode=True
    ):
        if chunk['type'] == 'reasoning_step':
            step = chunk['step']
            print(f"\n🧠 Step {step['step_number']}: {step['thought'][:100]}...")
            if step.get('action'):
                print(f"   🔧 Action: {step['action']}")

        elif chunk['type'] == 'tool_result':
            step = chunk['step']
            if step.get('observation'):
                print(f"   📋 Result: {step['observation'][:100]}...")

        elif chunk['type'] == 'final_response':
            print(f"\n🎯 Final Response:\n{chunk['response']}")
            print(f"📈 Total reasoning steps: {chunk['total_steps']}")

        elif chunk['type'] == 'error':
            print(f"\n❌ Error: {chunk['error']}")

    print("\n" + "="*50)


async def test_file_search():
    """Test file search tool"""
    print("🔍 Testing File Search...")

    client = ChatClient()

    print("\n💬 Sending: 'Can you search for Python files in the current directory?'")
    async for chunk in client.send_message("Can you search for Python files in the current directory?"):
        if chunk['type'] == 'content':
            print(chunk['content'], end='', flush=True)
        elif chunk['type'] == 'complete':
            print(f"\n✅ File search completed")

    print("\n" + "="*50)


async def test_session_history():
    """Test session history"""
    print("📜 Testing Session History...")

    client = ChatClient()

    # Send a few messages to build history
    messages = [
        "Hi, I'm testing the chat system.",
        "Can you remember my name is Alice?",
        "What's my name?"
    ]

    for message in messages:
        print(f"\n💬 Sending: '{message}'")
        async for chunk in client.send_message(message):
            if chunk['type'] == 'content':
                print(chunk['content'], end='', flush=True)
            elif chunk['type'] == 'complete':
                print()  # New line

    # Get session history
    history = await client.get_session_messages()
    print(f"\n📊 Session contains {history['count']} messages")
    for msg in history['messages'][-3:]:  # Show last 3 messages
        print(f"   {msg['role']}: {msg['content'][:50]}...")

    print("\n" + "="*50)


async def main():
    """Run all tests"""
    print("🚀 Starting Chat Backend Tests")
    print("="*50)

    try:
        await test_basic_chat()
        await test_calculator()
        await test_memory()
        await test_file_search()
        await test_session_history()
        await test_reasoning_mode()

        print("\n🎉 All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
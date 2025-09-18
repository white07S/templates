#!/usr/bin/env python3
"""Test script for memory functionality"""

import asyncio
import aiohttp
import json
import uuid


async def test_memory_system():
    base_url = "http://localhost:8000"
    user_id = "test_user"
    session_id = str(uuid.uuid4())

    async with aiohttp.ClientSession() as session:
        print(f"Testing with user_id: {user_id}, session_id: {session_id}")
        print("-" * 50)

        # Test 1: Send a message with information to remember
        print("\n1. Sending a message with information to remember...")
        async with session.post(
            f"{base_url}/chat",
            json={
                "session_id": session_id,
                "user_id": user_id,
                "message": "My name is Alice and I prefer short, concise answers. Remember this.",
                "temperature": 0.7,
                "max_tokens": 500,
                "reasoning_mode": False
            }
        ) as response:
            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode().replace('data: ', ''))
                        if data.get('type') == 'content':
                            print(data['content'], end='', flush=True)
                    except:
                        pass
        print("\n")

        # Wait a bit
        await asyncio.sleep(2)

        # Test 2: Ask about previous query
        print("\n2. Asking about previous query...")
        async with session.post(
            f"{base_url}/chat",
            json={
                "session_id": session_id,
                "user_id": user_id,
                "message": "What was my previous query?",
                "temperature": 0.7,
                "max_tokens": 500,
                "reasoning_mode": False
            }
        ) as response:
            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode().replace('data: ', ''))
                        if data.get('type') == 'content':
                            print(data['content'], end='', flush=True)
                    except:
                        pass
        print("\n")

        # Wait a bit
        await asyncio.sleep(2)

        # Test 3: Ask about remembered information
        print("\n3. Asking about remembered information...")
        async with session.post(
            f"{base_url}/chat",
            json={
                "session_id": session_id,
                "user_id": user_id,
                "message": "What's my name and what do I prefer?",
                "temperature": 0.7,
                "max_tokens": 500,
                "reasoning_mode": False
            }
        ) as response:
            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode().replace('data: ', ''))
                        if data.get('type') == 'content':
                            print(data['content'], end='', flush=True)
                    except:
                        pass
        print("\n")

        # Test 4: Check memory stats
        print("\n4. Checking memory stats...")
        async with session.get(f"{base_url}/memory/{user_id}/stats") as response:
            stats = await response.json()
            print(f"Memory stats: {json.dumps(stats, indent=2)}")

        # Test 5: Get session history
        print("\n5. Getting session history...")
        async with session.get(f"{base_url}/sessions/{user_id}/{session_id}") as response:
            history = await response.json()
            print(f"Session has {history['count']} messages")
            for msg in history['messages'][-3:]:  # Show last 3 messages
                print(f"  [{msg['role']}]: {msg['content'][:100]}...")


if __name__ == "__main__":
    print("Starting memory system test...")
    print("Make sure the server is running on http://localhost:8000")
    print("=" * 50)

    asyncio.run(test_memory_system())

    print("\n" + "=" * 50)
    print("Test completed!")
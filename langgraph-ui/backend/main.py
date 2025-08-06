"""
Enhanced client for testing the FastAPI streaming chat API.
Tests multiple streaming approaches and handles various edge cases.
"""

import json
import httpx
import asyncio
from typing import Optional
import sys
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8000"
SESSION_ID = None  # Will be set by the server if not provided

async def test_streaming_chat(message: str, session_id: Optional[str] = None):
    """Test the main streaming chat endpoint."""
    print(f"\n🚀 Streaming Request: {message}")
    print("-" * 50)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        payload = {
            "message": message,
            "session_id": session_id,
            "stream": True
        }
        
        try:
            async with client.stream(
                "POST",
                f"{API_BASE_URL}/chat/stream",
                json=payload,
            ) as response:
                response.raise_for_status()
                
                # Get session ID from headers
                session_id = response.headers.get("X-Session-Id")
                print(f"📌 Session ID: {session_id}\n")
                
                # Process streaming response
                buffer = ""
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            
                            if data["type"] == "content":
                                # Print content as it streams
                                print(data["data"], end="", flush=True)
                            
                            elif data["type"] == "tool_start":
                                print(f"\n🔧 Using tool: {data['tool']}", flush=True)
                            
                            elif data["type"] == "tool_end":
                                output = data.get('output', '')
                                if output:
                                    print(f"✅ Tool result: {output[:100]}...\n", flush=True)
                            
                            elif data["type"] == "done":
                                tools = data.get('tools_used', [])
                                if tools:
                                    print(f"\n\n✨ Complete! Tools used: {tools}")
                                else:
                                    print("\n\n✨ Complete!")
                            
                            elif data["type"] == "error":
                                print(f"\n❌ Error: {data['error']}")
                        
                        except json.JSONDecodeError as e:
                            # Log but don't crash on decode errors
                            print(f"\n⚠️ JSON decode error: {e} for line: {line[:100]}")
                
                return session_id
                
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error {e.response.status_code}: {e.response.text}")
        except httpx.TimeoutException:
            print("❌ Request timed out")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    return session_id

async def test_direct_streaming(message: str, session_id: Optional[str] = None):
    """Test direct vLLM streaming without LangGraph."""
    print(f"\n🔄 Direct vLLM Streaming: {message}")
    print("-" * 50)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        payload = {
            "message": message,
            "session_id": session_id,
            "stream": True
        }
        
        try:
            async with client.stream(
                "POST",
                f"{API_BASE_URL}/chat/direct-stream",
                json=payload,
            ) as response:
                response.raise_for_status()
                
                # Get session ID if provided
                session_id = response.headers.get("X-Session-Id")
                if session_id:
                    print(f"📌 Session ID: {session_id}\n")
                
                char_count = 0
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            
                            if data["type"] == "content":
                                # Print content as it streams
                                content = data["data"]
                                print(content, end="", flush=True)
                                char_count += len(content)
                            elif data["type"] == "done":
                                print(f"\n\n✨ Complete! ({char_count} characters streamed)")
                            elif data["type"] == "error":
                                print(f"\n❌ Error: {data['error']}")
                        except json.JSONDecodeError:
                            pass
                            
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error {e.response.status_code}: {e.response.text}")
        except httpx.TimeoutException:
            print("❌ Request timed out")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    return session_id

async def test_raw_vllm():
    """Test raw vLLM API to verify streaming works."""
    print(f"\n🔬 Testing Raw vLLM API")
    print("-" * 50)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        payload = {
            "message": "Tell me a very short story (2 sentences)",
            "stream": True
        }
        
        try:
            async with client.stream(
                "POST",
                f"{API_BASE_URL}/test/vllm-raw",
                json=payload,
            ) as response:
                response.raise_for_status()
                
                print("Raw vLLM response:")
                print("-" * 30)
                async for line in response.aiter_lines():
                    if line:
                        print(line)
                print("-" * 30)
                print("✅ Raw vLLM streaming is working!")
                            
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

async def test_regular_chat(message: str, session_id: Optional[str] = None):
    """Test the non-streaming chat endpoint."""
    print(f"\n📨 Regular Request: {message}")
    print("-" * 50)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        payload = {
            "message": message,
            "session_id": session_id,
            "stream": False
        }
        
        try:
            response = await client.post(
                f"{API_BASE_URL}/chat",
                json=payload,
            )
            response.raise_for_status()
            
            data = response.json()
            print(f"📌 Session ID: {data['session_id']}")
            print(f"🔧 Tools used: {data['tools_used']}")
            print(f"📝 Response: {data['response']}")
            print(f"⏰ Timestamp: {data['timestamp']}")
            
            return data['session_id']
            
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    return session_id

async def get_session_history(session_id: str):
    """Get the history of a session."""
    print(f"\n📜 Session History for: {session_id}")
    print("-" * 50)
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(
                f"{API_BASE_URL}/session/{session_id}/history",
            )
            response.raise_for_status()
            
            data = response.json()
            print(f"Total messages: {data['count']}")
            
            for i, msg in enumerate(data['messages'], 1):
                content = msg.get('content', 'N/A')
                if len(content) > 100:
                    content = content[:100] + "..."
                print(f"{i}. [{msg['type']}]: {content}")
            
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

async def clear_session(session_id: str):
    """Clear a session."""
    print(f"\n🗑️ Clearing session: {session_id}")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.delete(
                f"{API_BASE_URL}/session/{session_id}",
            )
            response.raise_for_status()
            
            data = response.json()
            print(f"✅ {data['message']}")
            
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

async def health_check():
    """Check API health."""
    print("\n🏥 Health Check")
    print("-" * 50)
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(f"{API_BASE_URL}/health")
            response.raise_for_status()
            
            data = response.json()
            print(f"✅ Status: {data['status']}")
            print(f"📊 Model: {data['config']['model']}")
            print(f"🔗 vLLM URL: {data['config']['vllm_url']}")
            print(f"⏰ Timestamp: {data['timestamp']}")
            
            return True
            
        except Exception as e:
            print(f"❌ API is not available: {str(e)}")
            return False

async def run_demo():
    """Run a complete demo of the API."""
    print("=" * 60)
    print("🎯 FastAPI LangGraph Streaming Demo")
    print("=" * 60)
    
    # Check health first
    if not await health_check():
        print("\n⚠️ Please ensure the API is running on http://localhost:8080")
        return
    
    # Test raw vLLM first to verify streaming works
    await test_raw_vllm()
    await asyncio.sleep(1)
    
    # Test queries
    test_queries = [
        ("What's the weather in New York?", "with_tools"),
        ("Calculate the square root of 144", "with_tools"),
        ("Tell me about LangGraph", "with_tools"),
        ("Hello! How are you?", "no_tools"),
        ("What's 25 * 4 and the weather in Tokyo?", "multiple_tools"),
    ]
    
    # Test streaming with tools
    print("\n" + "=" * 60)
    print("📡 STREAMING TESTS (with LangGraph)")
    print("=" * 60)
    
    session_id = None
    for query, test_type in test_queries[:3]:
        print(f"\n[Test Type: {test_type}]")
        session_id = await test_streaming_chat(query, session_id)
        await asyncio.sleep(1)
    
    # Test direct streaming
    print("\n" + "=" * 60)
    print("🔄 DIRECT vLLM STREAMING TEST (no tools)")
    print("=" * 60)
    
    await test_direct_streaming("Tell me a short story about a robot (3 sentences).", session_id)
    await asyncio.sleep(1)
    
    # Test non-streaming
    print("\n" + "=" * 60)
    print("📨 NON-STREAMING TESTS")
    print("=" * 60)
    
    for query, test_type in test_queries[3:]:
        print(f"\n[Test Type: {test_type}]")
        session_id = await test_regular_chat(query, session_id)
        await asyncio.sleep(1)
    
    # Get session history
    if session_id:
        await get_session_history(session_id)
    
    # Clear session
    if session_id:
        await clear_session(session_id)

async def interactive_mode():
    """Run in interactive mode."""
    print("\n" + "=" * 60)
    print("💬 Interactive Mode (type 'quit' to exit)")
    print("Commands: /stream, /direct, /regular, /test-raw, /history, /clear, /help")
    print("=" * 60)
    
    session_id = None
    mode = "stream"  # Default mode
    
    while True:
        try:
            user_input = input(f"\n[{mode} mode] You: ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            elif user_input == '/help':
                print("""
Commands:
  /stream    - Use streaming with LangGraph (supports tools)
  /direct    - Use direct vLLM streaming (no tools, true streaming)
  /regular   - Use non-streaming mode
  /test-raw  - Test raw vLLM API to verify streaming works
  /history   - Show session history
  /clear     - Clear current session
  /help      - Show this help
  quit       - Exit the application
                """)
            
            elif user_input == '/stream':
                mode = "stream"
                print("✅ Switched to streaming mode (with tools)")
            
            elif user_input == '/direct':
                mode = "direct"
                print("✅ Switched to direct vLLM streaming (no tools)")
            
            elif user_input == '/regular':
                mode = "regular"
                print("✅ Switched to non-streaming mode")
            
            elif user_input == '/test-raw':
                await test_raw_vllm()
            
            elif user_input == '/history':
                if session_id:
                    await get_session_history(session_id)
                else:
                    print("❌ No active session")
            
            elif user_input == '/clear':
                if session_id:
                    await clear_session(session_id)
                    session_id = None
                else:
                    print("❌ No active session")
            
            elif user_input:
                start_time = datetime.now()
                
                if mode == "stream":
                    session_id = await test_streaming_chat(user_input, session_id)
                elif mode == "direct":
                    session_id = await test_direct_streaming(user_input, session_id)
                elif mode == "regular":
                    session_id = await test_regular_chat(user_input, session_id)
                
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"\n⏱️ Response time: {elapsed:.2f}s")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

async def stress_test():
    """Run a stress test with concurrent requests."""
    print("\n" + "=" * 60)
    print("🔥 Stress Test - Concurrent Requests")
    print("=" * 60)
    
    if not await health_check():
        print("\n⚠️ API not available")
        return
    
    queries = [
        "What's the weather in London?",
        "Calculate 123 * 456",
        "Search for information about Python",
        "Hello, how are you?",
        "What's 2 + 2?",
    ]
    
    print(f"\n📊 Sending {len(queries)} concurrent requests...")
    
    start = datetime.now()
    tasks = [test_streaming_chat(q) for q in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = (datetime.now() - start).total_seconds()
    
    successful = sum(1 for r in results if not isinstance(r, Exception))
    print(f"\n✅ Completed: {successful}/{len(queries)} successful")
    print(f"⏱️ Total time: {elapsed:.2f}s")
    print(f"📈 Average: {elapsed/len(queries):.2f}s per request")

async def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            await run_demo()
        elif sys.argv[1] == "stress":
            await stress_test()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python client.py [demo|stress]")
    else:
        await interactive_mode()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
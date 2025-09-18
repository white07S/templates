import os
import asyncio
from typing import Optional, Dict, Any, AsyncGenerator, List
from openai import AsyncOpenAI
import httpx
from datetime import datetime, timezone


class LLMClient:
    """
    Unified LLM client that supports both OpenAI and vLLM backends
    """

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        timeout: float = 60.0
    ):
        self.provider = provider.lower()
        self.model = model
        self.timeout = timeout

        if self.provider == "openai":
            self.client = AsyncOpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                timeout=timeout
            )
        elif self.provider == "vllm":
            # vLLM is OpenAI-compatible
            self.client = AsyncOpenAI(
                api_key=api_key or "dummy-key",  # vLLM doesn't require real API key
                base_url=base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
                timeout=timeout
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def create_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ):
        """Create a chat completion"""
        try:
            completion_params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
                **kwargs
            }

            # Add tools if provided and supported
            if tools:
                completion_params["tools"] = tools
                if tool_choice:
                    completion_params["tool_choice"] = tool_choice

            response = await self.client.chat.completions.create(**completion_params)
            return response

        except Exception as e:
            raise Exception(f"LLM completion error: {str(e)}")

    async def create_streaming_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Create a streaming chat completion"""
        try:
            completion_params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
                **kwargs
            }

            # Add tools if provided
            if tools:
                completion_params["tools"] = tools
                if tool_choice:
                    completion_params["tool_choice"] = tool_choice

            stream = await self.client.chat.completions.create(**completion_params)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"Error: {str(e)}"

    async def health_check(self) -> Dict[str, Any]:
        """Check if the LLM service is healthy"""
        try:
            start_time = datetime.now(timezone.utc)

            # Simple test completion
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                temperature=0
            )

            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds()

            return {
                "status": "healthy",
                "provider": self.provider,
                "model": self.model,
                "response_time_seconds": response_time,
                "timestamp": end_time.isoformat()
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider,
                "model": self.model,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    @property
    def chat(self):
        """Provide access to the underlying client's chat interface"""
        return self.client.chat


class LLMClientManager:
    """
    Manager for multiple LLM clients with fallback support
    """

    def __init__(self):
        self.clients: Dict[str, LLMClient] = {}
        self.primary_client: Optional[str] = None
        self.fallback_clients: List[str] = []

    def add_client(
        self,
        name: str,
        provider: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        is_primary: bool = False
    ):
        """Add an LLM client"""
        client = LLMClient(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model
        )

        self.clients[name] = client

        if is_primary:
            self.primary_client = name
        else:
            self.fallback_clients.append(name)

    async def get_healthy_client(self) -> Optional[LLMClient]:
        """Get the first healthy client, starting with primary"""
        clients_to_check = []

        if self.primary_client:
            clients_to_check.append(self.primary_client)

        clients_to_check.extend(self.fallback_clients)

        for client_name in clients_to_check:
            if client_name in self.clients:
                client = self.clients[client_name]
                health = await client.health_check()

                if health["status"] == "healthy":
                    return client

        return None

    async def create_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ):
        """Create completion with automatic fallback"""
        client = await self.get_healthy_client()
        if not client:
            raise Exception("No healthy LLM clients available")

        return await client.create_completion(messages, **kwargs)

    async def create_streaming_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Create streaming completion with automatic fallback"""
        client = await self.get_healthy_client()
        if not client:
            raise Exception("No healthy LLM clients available")

        async for chunk in client.create_streaming_completion(messages, **kwargs):
            yield chunk

    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all clients"""
        results = {}

        for name, client in self.clients.items():
            results[name] = await client.health_check()

        return results


def create_default_llm_manager() -> LLMClientManager:
    """Create a default LLM manager with OpenAI and vLLM support"""
    manager = LLMClientManager()

    # Add OpenAI client as primary (if API key available)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        manager.add_client(
            name="openai",
            provider="openai",
            api_key=openai_key,
            model="gpt-4o-mini",
            is_primary=True
        )

    # Add vLLM as fallback
    vllm_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    vllm_model = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-32B-Instruct")

    manager.add_client(
        name="vllm",
        provider="vllm",
        base_url=vllm_url,
        model=vllm_model,
        is_primary=not openai_key  # Primary if no OpenAI key
    )

    return manager
"""
ReAct (Reasoning and Acting) Agent Implementation

This module implements a ReAct agent that can break down complex tasks into smaller steps,
reason about what actions to take, and execute tools in sequence or parallel as needed.

The ReAct pattern follows: Thought -> Action -> Observation -> Thought -> Action -> ...

Based on: https://www.ibm.com/think/topics/react-agent
"""

import json
import asyncio
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid

from openai import AsyncOpenAI
from config import settings
from tools import executor as tool_executor


class TaskType(Enum):
    """Types of tasks the agent can handle."""
    SIMPLE = "simple"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    COMPLEX = "complex"


class ExecutionMode(Enum):
    """How to execute multiple actions."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


@dataclass
class Action:
    """Represents a single action the agent can take."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    depends_on: List[str] = field(default_factory=list)  # Action IDs this depends on
    result: Optional[Any] = None
    error: Optional[str] = None
    completed: bool = False


@dataclass
class Thought:
    """Represents the agent's reasoning at each step."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    reasoning: str = ""
    confidence: float = 0.8
    next_actions: List[str] = field(default_factory=list)  # Action IDs


@dataclass
class Task:
    """Represents a decomposed task."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    task_type: TaskType = TaskType.SIMPLE
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    actions: List[Action] = field(default_factory=list)
    thoughts: List[Thought] = field(default_factory=list)
    completed: bool = False
    result: Optional[str] = None


class TaskPlanner:
    """Plans and decomposes complex tasks into manageable actions."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def analyze_query_complexity(self, query: str, context: List[Dict]) -> TaskType:
        """Analyze if a query is simple or requires complex decomposition."""
        if settings.openai_api_key == "sk-test-key-placeholder":
            # Simple heuristic for demo mode
            if any(word in query.lower() for word in ["and then", "after that", "also", "plus", "both"]):
                return TaskType.SEQUENTIAL
            elif any(word in query.lower() for word in ["compare", "analyze", "research", "investigate"]):
                return TaskType.PARALLEL
            else:
                return TaskType.SIMPLE

        analysis_prompt = f"""
        Analyze this user query to determine its complexity and execution requirements.

        Query: "{query}"

        Available tools: {[tool['function']['name'] for tool in tool_executor.get_available_tools()]}

        Classify as:
        - SIMPLE: Can be handled by a single tool or direct response
        - SEQUENTIAL: Requires multiple steps that depend on each other
        - PARALLEL: Requires multiple independent actions that can run simultaneously
        - COMPLEX: Requires both sequential and parallel operations

        Consider:
        - Does it ask for multiple unrelated things?
        - Do later steps depend on earlier results?
        - Can parts be done simultaneously?
        - Does it require reasoning across multiple domains?

        Respond with just the classification: SIMPLE, SEQUENTIAL, PARALLEL, or COMPLEX
        """

        try:
            response = await self.client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {"role": "system", "content": "You are a task analysis expert."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=50,
                temperature=0.3
            )

            classification = response.choices[0].message.content.strip().upper()
            if classification in [t.value.upper() for t in TaskType]:
                return TaskType(classification.lower())
            else:
                return TaskType.SIMPLE

        except Exception as e:
            print(f"Error analyzing query complexity: {e}")
            return TaskType.SIMPLE

    async def _simple_query_needs_tools(self, query: str, context: List[Dict]) -> bool:
        """Check if a simple query actually requires tool execution."""
        query_lower = query.lower()

        # Patterns that definitely need tools
        tool_patterns = [
            r'\b(calculate|compute|what is \d+)',
            r'\b(current time|what time)',
            r'\b(convert|change.*to)',
            r'\b(save|store|remember this)',
            r'\b(search|find|recall) memories',
            r'\b(random fact|generate)',
        ]

        for pattern in tool_patterns:
            if re.search(pattern, query_lower):
                return True

        # Check if it's asking about context
        context_patterns = [
            r'^(what|how|why|explain)',
            r'\b(mean|indicate|show)',
            r'^(can you|could you) (explain|clarify)',
        ]

        has_context_pattern = any(re.search(p, query_lower) for p in context_patterns)

        # If it has context patterns and there's recent context, probably doesn't need tools
        if has_context_pattern and len(context) > 2:
            return False

        # Default to needing tools for ambiguous cases
        return True

    async def decompose_task(self, query: str, task_type: TaskType, context: List[Dict]) -> Task:
        """Decompose a complex query into actionable steps."""
        # For simple tasks, check if tools are really needed
        if task_type == TaskType.SIMPLE:
            # Analyze if this simple query actually needs tools
            needs_tools = await self._simple_query_needs_tools(query, context)
            if not needs_tools:
                # Return empty task (no actions needed)
                return Task(
                    description=query,
                    task_type=task_type,
                    execution_mode=ExecutionMode.SEQUENTIAL,
                    actions=[]  # Empty actions list signals no tools needed
                )

        if settings.openai_api_key == "sk-test-key-placeholder":
            return self._decompose_task_fallback(query, task_type)

        available_tools = tool_executor.get_available_tools()
        tools_description = "\n".join([
            f"- {tool['function']['name']}: {tool['function']['description']}"
            for tool in available_tools
        ])

        decomposition_prompt = f"""
        Break down this complex query into specific, actionable steps using available tools.

        Query: "{query}"
        Task Type: {task_type.value}

        Available Tools:
        {tools_description}

        For each step, specify:
        1. What action to take (which tool to use)
        2. What arguments to pass
        3. Brief description of the step
        4. Dependencies (which previous steps must complete first)

        Determine execution strategy:
        - Start with PARALLEL execution for independent steps
        - Use dependencies ("depends_on") to enforce ordering when needed
        - Steps with no dependencies will run in parallel
        - Steps with dependencies will wait for their prerequisites

        IMPORTANT: Prefer parallel execution whenever possible. Only use sequential dependencies when:
        - One step needs the result of another step
        - There's a logical ordering requirement
        - Conditional logic depends on previous results

        Format as JSON:
        {{
            "execution_mode": "conditional",
            "steps": [
                {{
                    "id": "step_1",
                    "tool_name": "tool_name",
                    "arguments": {{"key": "value"}},
                    "description": "What this step does",
                    "depends_on": [] // empty for parallel execution, add step IDs for dependencies
                }},
                {{
                    "id": "step_2",
                    "tool_name": "another_tool",
                    "arguments": {{"key": "value"}},
                    "description": "What this step does",
                    "depends_on": ["step_1"] // if this depends on step_1's result
                }}
            ],
            "reasoning": "Explanation of the execution strategy"
        }}
        """

        try:
            response = await self.client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {"role": "system", "content": "You are an expert task decomposition agent. Think step by step."},
                    {"role": "user", "content": decomposition_prompt}
                ],
                max_tokens=1500,
                temperature=0.5,
                response_format={"type": "json_object"}
            )

            decomposition = json.loads(response.choices[0].message.content)
            return self._create_task_from_decomposition(query, task_type, decomposition)

        except Exception as e:
            print(f"Error decomposing task: {e}")
            return self._decompose_task_fallback(query, task_type)

    def _decompose_task_fallback(self, query: str, task_type: TaskType) -> Task:
        """Fallback decomposition when LLM is not available."""
        task = Task(description=query, task_type=task_type)

        # Simple heuristic-based decomposition
        if "time" in query.lower():
            task.actions.append(Action(
                tool_name="get_current_time",
                arguments={"timezone": "UTC"},
                description="Get current time"
            ))

        if any(word in query.lower() for word in ["calculate", "math", "compute"]):
            # Extract potential math expression (very basic)
            words = query.split()
            for i, word in enumerate(words):
                if any(op in word for op in ["+", "-", "*", "/", "="]):
                    expr = " ".join(words[max(0, i-2):i+3])
                    task.actions.append(Action(
                        tool_name="calculate_math",
                        arguments={"expression": expr},
                        description="Perform calculation"
                    ))
                    break

        if any(word in query.lower() for word in ["memory", "remember", "recall"]):
            task.actions.append(Action(
                tool_name="search_memories",
                arguments={"query": query, "user_id": "current_user", "limit": 5},
                description="Search user memories"
            ))

        if not task.actions:
            # Default action for unknown queries
            task.actions.append(Action(
                tool_name="generate_random_fact",
                arguments={"category": "general"},
                description="Generate a random fact"
            ))

        return task

    def _create_task_from_decomposition(self, query: str, task_type: TaskType, decomposition: Dict) -> Task:
        """Create a Task object from LLM decomposition."""
        task = Task(
            description=query,
            task_type=task_type,
            execution_mode=ExecutionMode(decomposition.get("execution_mode", "sequential"))
        )

        # Create thoughts
        if "reasoning" in decomposition:
            task.thoughts.append(Thought(
                content=f"Task decomposition: {decomposition['reasoning']}",
                reasoning=decomposition['reasoning']
            ))

        # Create actions
        for i, step in enumerate(decomposition.get("steps", [])):
            # Use provided ID or generate one
            action_id = step.get("id", f"step_{i}")

            action = Action(
                id=action_id,
                tool_name=step.get("tool_name", ""),
                arguments=step.get("arguments", {}),
                description=step.get("description", ""),
                depends_on=step.get("depends_on", [])
            )
            task.actions.append(action)

        return task


class ExecutionEngine:
    """Executes tasks with support for sequential and parallel execution."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def execute_task(self, task: Task, user_id: str) -> Task:
        """Execute a complete task."""
        print(f"🎯 Executing task: {task.description}")
        print(f"📋 Task type: {task.task_type.value}, Execution mode: {task.execution_mode.value}")
        print(f"🔧 Actions to execute: {len(task.actions)}")

        if task.execution_mode == ExecutionMode.PARALLEL:
            await self._execute_parallel(task, user_id)
        elif task.execution_mode == ExecutionMode.CONDITIONAL:
            await self._execute_conditional(task, user_id)
        else:  # SEQUENTIAL
            await self._execute_sequential(task, user_id)

        task.completed = True
        task.result = await self._synthesize_results(task)

        print(f"✅ Task completed: {task.description}")
        return task

    async def _execute_sequential(self, task: Task, user_id: str):
        """Execute actions sequentially."""
        for action in task.actions:
            print(f"🔄 Executing: {action.description}")
            await self._execute_action(action, user_id)

    async def _execute_parallel(self, task: Task, user_id: str):
        """Execute actions in parallel."""
        # Group actions by dependencies
        independent_actions = [a for a in task.actions if not a.depends_on]
        dependent_actions = [a for a in task.actions if a.depends_on]

        # Execute independent actions first
        if independent_actions:
            print(f"⚡ Executing {len(independent_actions)} independent actions in parallel")
            await asyncio.gather(*[
                self._execute_action(action, user_id)
                for action in independent_actions
            ])

        # Execute dependent actions after their dependencies
        while dependent_actions:
            ready_actions = []
            for action in dependent_actions:
                deps_completed = []
                for dep_id in action.depends_on:
                    dep_action = self._find_action_by_id(task.actions, dep_id)
                    deps_completed.append(dep_action and dep_action.completed)

                if all(deps_completed):
                    ready_actions.append(action)

            if not ready_actions:
                print("⚠️ Circular dependency detected, executing remaining sequentially")
                for action in dependent_actions:
                    await self._execute_action(action, user_id)
                break

            if len(ready_actions) > 1:
                print(f"⚡ Executing {len(ready_actions)} dependent actions in parallel")
            else:
                print(f"🔄 Executing dependent action: {ready_actions[0].description}")

            await asyncio.gather(*[
                self._execute_action(action, user_id)
                for action in ready_actions
            ])

            for action in ready_actions:
                dependent_actions.remove(action)

    async def _execute_conditional(self, task: Task, user_id: str):
        """Execute actions conditionally based on results."""
        for action in task.actions:
            # Check if dependencies are met
            can_execute = True
            for dep_id in action.depends_on:
                dep_action = self._find_action_by_id(task.actions, dep_id)
                if not dep_action or not dep_action.completed:
                    can_execute = False
                    break

            if can_execute:
                print(f"🔄 Executing: {action.description}")
                await self._execute_action(action, user_id)
            else:
                print(f"⏸️  Skipping: {action.description} (dependencies not met)")

    async def _execute_action(self, action: Action, user_id: str):
        """Execute a single action."""
        try:
            # Add user_id to arguments if the tool expects it
            if action.tool_name in ["search_memories", "get_memory_stats", "save_user_preference"]:
                action.arguments["user_id"] = user_id

            result = await tool_executor.registry.execute_tool(
                action.tool_name,
                action.arguments
            )

            action.result = result
            action.completed = True
            print(f"✅ Completed: {action.description}")

        except Exception as e:
            action.error = str(e)
            action.completed = True
            print(f"❌ Failed: {action.description} - {e}")

    def _find_action_by_id(self, actions: List[Action], action_id: str) -> Optional[Action]:
        """Find action by ID."""
        return next((a for a in actions if a.id == action_id), None)

    async def _synthesize_results(self, task: Task) -> str:
        """Synthesize results from all actions into a coherent response."""
        if not task.actions:
            return "No actions were executed."

        successful_results = []
        failed_actions = []

        for action in task.actions:
            if action.completed and not action.error:
                result_text = self._format_action_result(action)
                successful_results.append(f"{action.description}: {result_text}")
            elif action.error:
                failed_actions.append(f"{action.description}: {action.error}")

        # Build response
        response_parts = []

        if successful_results:
            if len(successful_results) == 1:
                response_parts.append(successful_results[0])
            else:
                response_parts.append("Here are the results:")
                for result in successful_results:
                    response_parts.append(f"• {result}")

        if failed_actions:
            response_parts.append("\nSome actions encountered issues:")
            for failure in failed_actions:
                response_parts.append(f"• {failure}")

        return "\n".join(response_parts) if response_parts else "Task completed with no specific results."

    def _format_action_result(self, action: Action) -> str:
        """Format action result for human readability."""
        if not action.result:
            return "No result"

        if isinstance(action.result, dict):
            if action.tool_name == "get_current_time":
                return f"Current time is {action.result.get('formatted', 'unknown')}"
            elif action.tool_name == "calculate_math":
                return f"{action.result.get('expression', '')} = {action.result.get('result', 'error')}"
            elif action.tool_name == "generate_random_fact":
                return action.result.get('fact', 'No fact available')
            elif action.tool_name == "search_memories":
                memories = action.result.get('memories', [])
                if memories:
                    return f"Found {len(memories)} relevant memories"
                else:
                    return "No relevant memories found"
            elif action.tool_name == "get_memory_stats":
                total = action.result.get('total_memories', 0)
                return f"You have {total} stored memories"
            else:
                # Generic dict formatting
                return str(action.result)
        else:
            return str(action.result)


class ReActAgent:
    """Main ReAct agent that coordinates planning and execution."""

    def __init__(self):
        self.planner = TaskPlanner()
        self.executor = ExecutionEngine()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def process_query(self, query: str, user_id: str, context: List[Dict], check_tools_needed: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        Process a query using the ReAct pattern with intelligent tool usage.

        Args:
            query: The user's query
            user_id: The user ID
            context: Conversation context
            check_tools_needed: Whether to check if tools are actually needed

        Returns:
            Tuple of (response_text, metadata)
        """
        start_time = datetime.utcnow()

        print(f"🤖 ReAct Agent processing: {query}")

        # Step 1: Check if tools are actually needed (new logic)
        if check_tools_needed:
            tools_needed = await self._check_if_tools_needed(query, context)
            if not tools_needed:
                print(f"💬 ReAct Agent: No tools needed, generating direct response")
                # Generate response without tools
                direct_response = await self._generate_direct_response(query, context)
                metadata = {
                    "agent_type": "react",
                    "task_type": "direct_response",
                    "tools_used": False,
                    "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
                }
                return direct_response, metadata

        # Step 2: Analyze complexity
        task_type = await self.planner.analyze_query_complexity(query, context)
        print(f"📊 Query complexity: {task_type.value}")

        # Step 3: Decompose if needed
        task = await self.planner.decompose_task(query, task_type, context)

        # Step 4: Check if decomposition resulted in no tools needed
        if not task.actions:
            print(f"💬 ReAct Agent: No actions needed after decomposition")
            direct_response = await self._generate_direct_response(query, context)
            metadata = {
                "agent_type": "react",
                "task_type": task_type.value,
                "tools_used": False,
                "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
            }
            return direct_response, metadata

        # Step 5: Execute task
        completed_task = await self.executor.execute_task(task, user_id)

        # Step 6: Generate final response
        final_response = await self._generate_final_response(query, completed_task, context)

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Collect tool results for metadata
        tool_results = {}
        for action in task.actions:
            if action.completed and action.result and not action.error:
                tool_results[action.tool_name] = action.result

        metadata = {
            "agent_type": "react",
            "task_type": task_type.value,
            "execution_mode": task.execution_mode.value,
            "actions_executed": len(task.actions),
            "actions_successful": sum(1 for a in task.actions if a.completed and not a.error),
            "processing_time_ms": processing_time,
            "task_id": task.id,
            "tools_used": True,
            "tool_results": tool_results
        }

        return final_response, metadata

    async def _check_if_tools_needed(self, query: str, context: List[Dict]) -> bool:
        """
        Check if the query actually needs tools or can be answered directly from context.
        """
        if settings.openai_api_key == "sk-test-key-placeholder":
            return True  # Default to needing tools in test mode

        check_prompt = f"""
        Analyze this query to determine if it requires external tool execution or can be answered from the conversation context.

        Query: "{query}"

        Available conversation context: {len(context)} messages

        Consider:
        1. Is this a follow-up question about previous results?
        2. Can this be answered from information already in the conversation?
        3. Does this require new data from tools (calculations, time, memory search, etc.)?
        4. Is this asking for clarification or explanation of existing information?

        Respond with ONLY "TOOLS_NEEDED" or "NO_TOOLS_NEEDED" and nothing else.
        """

        try:
            response = await self.client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {"role": "system", "content": "You are a query analyzer. Respond only with TOOLS_NEEDED or NO_TOOLS_NEEDED."},
                    {"role": "user", "content": check_prompt}
                ],
                max_tokens=10,
                temperature=0
            )

            result = response.choices[0].message.content.strip().upper()
            return "TOOLS_NEEDED" in result

        except Exception as e:
            print(f"Error checking if tools needed: {e}")
            return True  # Default to needing tools on error

    async def _generate_direct_response(self, query: str, context: List[Dict]) -> str:
        """
        Generate a direct response without using tools.
        """
        if settings.openai_api_key == "sk-test-key-placeholder":
            return f"Based on the context, here's my response to: {query}"

        try:
            # Add the user's query to context
            full_context = context + [{"role": "user", "content": query}]

            response = await self.client.chat.completions.create(
                model=settings.chat_model,
                messages=full_context,
                max_tokens=800,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error generating direct response: {e}")
            return f"I understand your question about: {query}. Based on the context provided, let me help you with that."

    async def _generate_final_response(self, query: str, task: Task, context: List[Dict]) -> str:
        """Generate a natural response based on the task execution results."""
        if not task.result:
            return "I wasn't able to complete your request successfully."

        if settings.openai_api_key == "sk-test-key-placeholder":
            return f"Based on my analysis, here's what I found:\n\n{task.result}"

        # Use LLM to generate a natural response
        synthesis_prompt = f"""
        Generate a natural, conversational response based on the task execution results.

        Original query: "{query}"
        Task results: {task.result}

        Guidelines:
        - Be conversational and helpful
        - Integrate the results naturally
        - Don't mention technical details about task execution
        - If multiple results, present them coherently
        - If there were errors, acknowledge them gracefully

        Response:
        """

        try:
            response = await self.client.chat.completions.create(
                model=settings.chat_model,
                messages=context + [
                    {"role": "system", "content": "You are a helpful assistant that presents task results in a natural, conversational way."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error generating final response: {e}")
            return f"Here's what I found:\n\n{task.result}"
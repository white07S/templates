import json
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
from enum import Enum

from ..models.schemas import ReasoningStep, ToolCall, ToolResult
from ..tools.base_tools import ToolRegistry


class ReasoningState(str, Enum):
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETE = "complete"


class ReActAgent:
    """
    ReAct (Reasoning + Acting) agent that implements iterative reasoning
    with tool usage for complex problem solving.
    """

    def __init__(self, llm_client, tool_registry: ToolRegistry, max_iterations: int = 10):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations

    async def reason_and_act(
        self,
        task: str,
        context: str = "",
        user_id: str = "default",
        session_id: str = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute ReAct reasoning loop for a given task
        """
        reasoning_steps = []
        current_step = 1
        state = ReasoningState.THINKING

        # Initial system prompt for reasoning
        system_prompt = self._get_system_prompt()

        # Build conversation context
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task}\n\nContext: {context}"}
        ]

        while current_step <= self.max_iterations and state != ReasoningState.COMPLETE:
            try:
                if state == ReasoningState.THINKING:
                    # Thinking phase
                    thought_step = await self._think(messages, current_step, reasoning_steps)
                    reasoning_steps.append(thought_step)

                    yield {
                        "type": "reasoning_step",
                        "step": thought_step.model_dump(),
                        "state": state.value
                    }

                    # Decide next action based on thought
                    if thought_step.action:
                        state = ReasoningState.ACTING
                    else:
                        state = ReasoningState.COMPLETE

                elif state == ReasoningState.ACTING:
                    # Acting phase - execute the planned action
                    last_step = reasoning_steps[-1]
                    observation = await self._act(last_step, user_id, session_id)

                    # Update the last step with observation
                    last_step.observation = observation
                    reasoning_steps[-1] = last_step

                    yield {
                        "type": "tool_result",
                        "step": last_step.model_dump(),
                        "state": state.value
                    }

                    # Add observation to conversation
                    messages.append({
                        "role": "assistant",
                        "content": f"Thought: {last_step.thought}\nAction: {last_step.action}\nObservation: {observation}"
                    })

                    state = ReasoningState.THINKING
                    current_step += 1

            except Exception as e:
                error_step = ReasoningStep(
                    step_number=current_step,
                    thought=f"Error occurred: {str(e)}",
                    observation=f"Failed to execute step: {str(e)}"
                )
                reasoning_steps.append(error_step)

                yield {
                    "type": "error",
                    "step": error_step.model_dump(),
                    "error": str(e)
                }
                break

        # Generate final response
        final_response = await self._generate_final_response(task, reasoning_steps)

        yield {
            "type": "final_response",
            "response": final_response,
            "reasoning_steps": [step.model_dump() for step in reasoning_steps],
            "total_steps": len(reasoning_steps)
        }

    async def _think(
        self,
        messages: List[Dict[str, str]],
        step_number: int,
        previous_steps: List[ReasoningStep]
    ) -> ReasoningStep:
        """Generate reasoning step"""

        # Add reasoning context
        if previous_steps:
            reasoning_context = "\n".join([
                f"Step {step.step_number}: {step.thought}"
                + (f" -> Action: {step.action}" if step.action else "")
                + (f" -> Result: {step.observation}" if step.observation else "")
                for step in previous_steps
            ])
            messages.append({
                "role": "system",
                "content": f"Previous reasoning:\n{reasoning_context}\n\nContinue reasoning for step {step_number}:"
            })

        # Get LLM response for reasoning
        response = await self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        thought_content = response.choices[0].message.content.strip()

        # Parse thought and potential action
        action, action_input = self._parse_action_from_thought(thought_content)

        return ReasoningStep(
            step_number=step_number,
            thought=thought_content,
            action=action,
            action_input=action_input
        )

    def _parse_action_from_thought(self, thought: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Parse action from thought text"""
        thought_lower = thought.lower()

        # Look for action indicators
        action_keywords = {
            "calculate": "calculator",
            "search": "file_search",
            "time": "time_weather",
            "weather": "time_weather",
            "memory": "memory_tool",
            "note": "note_taking",
            "remember": "memory_tool"
        }

        for keyword, tool_name in action_keywords.items():
            if keyword in thought_lower and tool_name in self.tool_registry.tools:
                # Extract action parameters based on context
                action_input = self._extract_action_parameters(thought, tool_name)
                return tool_name, action_input

        return None, None

    def _extract_action_parameters(self, thought: str, tool_name: str) -> Dict[str, Any]:
        """Extract parameters for tool execution from thought"""
        if tool_name == "calculator":
            # Look for mathematical expressions
            import re
            math_pattern = r'[\d+\-*/().\s]+'
            matches = re.findall(math_pattern, thought)
            if matches:
                # Take the longest match as the expression
                expression = max(matches, key=len).strip()
                return {"expression": expression}

        elif tool_name == "file_search":
            # Extract search pattern
            words = thought.split()
            search_terms = [word for word in words if not word.lower() in ["search", "find", "file", "files"]]
            if search_terms:
                return {"pattern": "*" + search_terms[0] + "*"}

        elif tool_name == "time_weather":
            return {
                "include_time": "time" in thought.lower(),
                "include_weather": "weather" in thought.lower()
            }

        elif tool_name == "memory_tool":
            if "remember" in thought.lower() or "save" in thought.lower():
                # Extract content to remember
                content = thought.replace("remember", "").replace("save", "").strip()
                return {"action": "add", "content": content, "importance": 0.7}
            else:
                # Search memory
                query = thought.replace("memory", "").replace("recall", "").strip()
                return {"action": "search", "query": query}

        elif tool_name == "note_taking":
            if "create" in thought.lower() or "write" in thought.lower():
                return {"action": "create", "title": "Quick Note", "content": thought}
            else:
                return {"action": "list"}

        return {}

    async def _act(
        self,
        step: ReasoningStep,
        user_id: str,
        session_id: Optional[str]
    ) -> str:
        """Execute the action specified in the reasoning step"""
        if not step.action:
            return "No action to execute"

        try:
            # Add user context to action input
            action_input = step.action_input or {}
            if step.action == "memory_tool":
                action_input["user_id"] = user_id
                action_input["session_id"] = session_id

            # Execute the tool
            result = await self.tool_registry.execute_tool(step.action, **action_input)

            # Format the result
            if "error" in result:
                return f"Error: {result['error']}"
            else:
                return self._format_tool_result(step.action, result)

        except Exception as e:
            return f"Action execution failed: {str(e)}"

    def _format_tool_result(self, tool_name: str, result: Dict[str, Any]) -> str:
        """Format tool execution result for observation"""
        if tool_name == "calculator":
            if "result" in result:
                return f"Calculation result: {result['result']}"

        elif tool_name == "file_search":
            if "results" in result:
                count = result["count"]
                if count == 0:
                    return "No files found matching the pattern"
                else:
                    files = [r["name"] for r in result["results"][:5]]  # Show first 5
                    return f"Found {count} files: {', '.join(files)}"

        elif tool_name == "time_weather":
            parts = []
            if "current_time" in result:
                parts.append(f"Current time: {result['current_time']['formatted']}")
            if "weather" in result:
                weather = result["weather"]
                parts.append(f"Weather in {weather['location']}: {weather['condition']}, {weather['temperature']}")
            return "; ".join(parts)

        elif tool_name == "memory_tool":
            if "results" in result:
                count = result["count"]
                return f"Found {count} relevant memories"
            elif "success" in result:
                return "Memory saved successfully"
            elif "stats" in result:
                stats = result["stats"]
                return f"Memory stats: {stats['total']} total memories"

        elif tool_name == "note_taking":
            action = result.get("action", "unknown")
            if action == "create":
                return f"Note '{result.get('title')}' created successfully"
            elif action == "list":
                count = result.get("count", 0)
                return f"Found {count} notes"

        # Default formatting
        return str(result)

    async def _generate_final_response(
        self,
        original_task: str,
        reasoning_steps: List[ReasoningStep]
    ) -> str:
        """Generate final response based on reasoning steps"""

        # Summarize the reasoning process
        step_summary = []
        for step in reasoning_steps:
            summary = f"Step {step.step_number}: {step.thought[:100]}..."
            if step.action:
                summary += f" (Used tool: {step.action})"
            step_summary.append(summary)

        # Create final response prompt
        final_prompt = f"""
Based on the following reasoning process for the task "{original_task}":

{chr(10).join(step_summary)}

Please provide a clear, concise final answer that:
1. Directly addresses the original task
2. Incorporates key findings from the reasoning steps
3. Is actionable and helpful

Final Answer:"""

        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.3,
                max_tokens=300
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Completed reasoning process with {len(reasoning_steps)} steps. Task: {original_task}"

    def _get_system_prompt(self) -> str:
        """Get system prompt for ReAct reasoning"""
        available_tools = ", ".join(self.tool_registry.list_tools())

        return f"""You are a reasoning agent that uses a "Thought-Action-Observation" approach to solve problems.

For each step:
1. Think through the problem step by step
2. If you need more information or need to perform an action, specify what tool to use
3. Based on the observation, continue reasoning

Available tools: {available_tools}

Guidelines:
- Break down complex problems into smaller steps
- Use tools when you need external information or computation
- Be explicit about your reasoning
- Continue until you have enough information to provide a complete answer

When you think an action is needed, mention the specific tool name in your thought.
Examples:
- "I need to calculate 15 * 23, so I'll use the calculator"
- "Let me search for files with .py extension using file search"
- "I should save this important information to memory"

Think step by step and be thorough in your reasoning."""
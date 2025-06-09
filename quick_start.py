#!/usr/bin/env python3
"""
Quick Start Script for AI Backend
Run this script to quickly test both Normal AI and Agentic AI workflows
"""

import os
import sys
import time
import requests
import json
from example_agent_configs import MATH_TUTORING_WORKFLOW, MATH_OPTIMIZATION_WORKFLOW

# Configuration
BASE_URL = "http://localhost:8000"
COLORS = {
    'GREEN': '\033[92m',
    'RED': '\033[91m',
    'BLUE': '\033[94m',
    'YELLOW': '\033[93m',
    'PURPLE': '\033[95m',
    'CYAN': '\033[96m',
    'END': '\033[0m',
    'BOLD': '\033[1m'
}

def print_colored(text, color='END'):
    """Print colored text"""
    print(f"{COLORS.get(color, '')}{text}{COLORS['END']}")

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print_colored(f"🚀 {title}", 'BOLD')
    print("="*60)

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print_colored("✅ API is running and healthy!", 'GREEN')
            print_colored(f"Memory v2 available: {result.get('memory_v2_available', 'Unknown')}", 'CYAN')
            print_colored(f"Active workflows: {result.get('active_workflows', 0)}", 'CYAN')
            return True
        else:
            print_colored(f"❌ API returned status {response.status_code}", 'RED')
            return False
    except requests.exceptions.RequestException as e:
        print_colored(f"❌ Cannot connect to API: {e}", 'RED')
        print_colored("Make sure to run: python app.py", 'YELLOW')
        return False

def test_normal_ai_quick():
    """Quick test of Normal AI"""
    print_header("Testing Normal AI - Math Tutor")
    
    request_data = {
        "persona": "You are a friendly math tutor who explains concepts clearly and encouragingly.",
        "task": "Solve this step by step: If a triangle has sides of length 3, 4, and 5, is it a right triangle? Explain using the Pythagorean theorem."
    }
    
    try:
        print_colored("📤 Sending request to Normal AI...", 'BLUE')
        response = requests.post(f"{BASE_URL}/api/v1/normal-ai", json=request_data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print_colored("✅ Success!", 'GREEN')
            print_colored(f"Session ID: {result['session_id']}", 'CYAN')
            print_colored("📝 Response:", 'PURPLE')
            print(result['response'])
            return result['session_id']
        else:
            print_colored(f"❌ Error: {response.status_code}", 'RED')
            print(response.text)
            return None
            
    except Exception as e:
        print_colored(f"❌ Request failed: {e}", 'RED')
        return None

def wait_for_agent_completion(session_id, expected_agent_index, max_wait=120):
    """Wait for agent to complete before proceeding"""
    print_colored(f"⏳ Waiting for agent {expected_agent_index} to complete...", 'YELLOW')
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{BASE_URL}/api/v1/agentic-ai/status/{session_id}", timeout=10)
            if response.status_code == 200:
                status = response.json()
                completion_status = status.get('completion_status', {})
                agent_key = f"agent_{expected_agent_index}"
                
                if completion_status.get(agent_key, False):
                    print_colored(f"✅ Agent {expected_agent_index} completed!", 'GREEN')
                    return True
                else:
                    print_colored(f"⏳ Agent {expected_agent_index} still working... ({int(time.time() - start_time)}s)", 'YELLOW')
                    time.sleep(3)
            else:
                print_colored(f"❌ Status check failed: {response.status_code}", 'RED')
                time.sleep(2)
                
        except Exception as e:
            print_colored(f"⚠️ Status check error: {e}", 'YELLOW')
            time.sleep(2)
    
    print_colored(f"⚠️ Timeout waiting for agent {expected_agent_index}", 'RED')
    return False

def test_agentic_ai_simple():
    """Quick test of Agentic AI with simple math workflow"""
    print_header("Testing Agentic AI - Simple Math Collaboration")
    
    # Use the math tutoring workflow from configs
    request_data = {
        "agents": MATH_TUTORING_WORKFLOW["agents"],
        "initial_task": "Solve this algebra problem step by step and explain the concepts: 2x² - 8x + 6 = 0"
    }
    
    try:
        print_colored("📤 Starting Agentic Workflow...", 'BLUE')
        response = requests.post(f"{BASE_URL}/api/v1/agentic-ai", json=request_data, timeout=90)
        
        if response.status_code != 200:
            print_colored(f"❌ Failed to start workflow: {response.status_code}", 'RED')
            print(response.text)
            return None
        
        result = response.json()
        session_id = result['session_id']
        print_colored(f"✅ Workflow started! Session: {session_id}", 'GREEN')
        print_colored(f"🤖 Agent 1 ({result['current_agent_name']}):", 'PURPLE')
        print(result['response'][:400] + "..." if len(result['response']) > 400 else result['response'])
        
        # Wait for first agent to complete fully
        if not wait_for_agent_completion(session_id, 0):
            return None
        
        # Move through the remaining agents with proper waiting
        for i in range(1, len(MATH_TUTORING_WORKFLOW["agents"])):
            print_colored(f"\n➡️ Moving to Agent {i+1}...", 'YELLOW')
            
            next_response = requests.post(f"{BASE_URL}/api/v1/agentic-ai/next?session_id={session_id}", timeout=90)
            
            if next_response.status_code == 200:
                next_result = next_response.json()
                print_colored(f"🤖 Agent {i+1} ({next_result['current_agent_name']}):", 'PURPLE')
                print(next_result['response'][:400] + "..." if len(next_result['response']) > 400 else next_result['response'])
                
                # Wait for this agent to complete if not the last one
                if i < len(MATH_TUTORING_WORKFLOW["agents"]) - 1:
                    if not wait_for_agent_completion(session_id, i):
                        break
            else:
                print_colored(f"❌ Failed to move to next agent: {next_response.status_code}", 'RED')
                print(next_response.text)
                break
        
        return session_id
        
    except Exception as e:
        print_colored(f"❌ Workflow failed: {e}", 'RED')
        return None

def test_feedback_mechanism():
    """Test the feedback mechanism with a proper single-agent workflow"""
    print_header("Testing Feedback Mechanism")
    
    # Create a single-agent workflow for feedback testing
    request_data = {
        "agents": [
            {
                "name": "Math_Calculator",
                "persona": "You are a helpful calculator who solves math problems step by step with clear explanations.",
                "task": "Solve mathematical problems with detailed explanations, showing all work and reasoning."
            },
            {
                "name": "Concept_Explainer",
                "persona": "You are an educational specialist who explains mathematical concepts clearly.",
                "task": "Explain the underlying concepts and provide additional examples."
            },
            {
                "name": "Practice_Generator",
                "persona": "You are a practice problem generator who creates similar problems for reinforcement.",
                "task": "Generate practice problems and solutions based on the previous work."
            }
        ],
        "initial_task": "Calculate 15% of 240 and explain how to calculate percentages in general."
    }
    
    try:
        print_colored("📤 Starting new workflow for feedback test...", 'BLUE')
        response = requests.post(f"{BASE_URL}/api/v1/agentic-ai", json=request_data, timeout=60)
        
        if response.status_code != 200:
            print_colored(f"❌ Failed to start feedback test workflow: {response.status_code}", 'RED')
            print(response.text)
            return
        
        result = response.json()
        session_id = result['session_id']
        print_colored(f"✅ Initial response from {result['current_agent_name']}:", 'GREEN')
        print(result['response'][:300] + "..." if len(result['response']) > 300 else result['response'])
        
        # Wait for agent to complete
        if not wait_for_agent_completion(session_id, 0):
            return
        
        # Provide feedback
        feedback_data = {
            "session_id": session_id,
            "agent_index": 0,
            "feedback": "Great explanation! Can you also show me how to calculate 15% using the decimal method (0.15 × 240) and explain when to use different percentage calculation methods?"
        }
        
        print_colored("\n💬 Providing feedback...", 'YELLOW')
        feedback_response = requests.post(f"{BASE_URL}/api/v1/agentic-ai/feedback", json=feedback_data, timeout=60)
        
        if feedback_response.status_code == 200:
            feedback_result = feedback_response.json()
            print_colored("✅ Feedback processed! Updated response:", 'GREEN')
            print(feedback_result['response'][:300] + "..." if len(feedback_result['response']) > 300 else feedback_result['response'])
            
            # Continue to next agent
            print_colored("\n➡️ Moving to next agent...", 'YELLOW')
            next_response = requests.post(f"{BASE_URL}/api/v1/agentic-ai/next?session_id={session_id}", timeout=60)
            if next_response.status_code == 200:
                next_result = next_response.json()
                print_colored(f"🤖 {next_result['current_agent_name']} response:", 'PURPLE')
                print(next_result['response'][:300] + "..." if len(next_result['response']) > 300 else next_result['response'])
        else:
            print_colored(f"❌ Feedback failed: {feedback_response.status_code}", 'RED')
            print(feedback_response.text)
            
    except Exception as e:
        print_colored(f"❌ Feedback test failed: {e}", 'RED')

def demonstrate_complex_workflow():
    """Demonstrate a complex multi-agent workflow with proper completion checking"""
    print_header("Advanced Demo - Complex Business Problem")
    
    # Simplified version of the optimization workflow
    complex_agents = [
        {
            "name": "Business_Analyst",
            "persona": "You are a senior business analyst who breaks down complex business problems into manageable components and identifies key metrics.",
            "task": "Analyze the business problem comprehensively, identify key factors, constraints, objectives, and provide detailed financial analysis with calculations."
        },
        {
            "name": "Solution_Architect", 
            "persona": "You are a strategic solution architect who designs comprehensive solutions to business challenges using data-driven approaches.",
            "task": "Design practical, implementable solutions based on the business analysis, considering feasibility, resources, and expected outcomes with specific recommendations."
        },
        {
            "name": "Implementation_Planner",
            "persona": "You are an implementation specialist who creates detailed, actionable plans with timelines and success metrics.",
            "task": "Create a detailed implementation roadmap with clear steps, timelines, resource requirements, and measurable success criteria."
        }
    ]
    
    business_problem = """
    A small e-commerce company is struggling with the following metrics:
    
    Current Performance:
    - Monthly revenue: $50,000
    - Cart abandonment rate: 30%
    - Customer acquisition cost (CAC): $45 per customer
    - Average order value (AOV): $65
    - Monthly customer churn: 15%
    - Website conversion rate: 2.5%
    - Total monthly visitors: 20,000
    - Customer lifetime value (CLV): $180
    
    Business Goal:
    - Increase monthly revenue to $100,000 within 6 months
    - Improve profitability and sustainability
    
    Constraints:
    - Marketing budget: $15,000/month
    - Development budget: $10,000/month
    - Small team of 8 people
    
    Question: What comprehensive strategy should they implement to achieve their revenue goal?
    """
    
    request_data = {
        "agents": complex_agents,
        "initial_task": business_problem
    }
    
    try:
        print_colored("📤 Starting complex business analysis workflow...", 'BLUE')
        response = requests.post(f"{BASE_URL}/api/v1/agentic-ai", json=request_data, timeout=120)
        
        if response.status_code != 200:
            print_colored(f"❌ Failed to start complex workflow: {response.status_code}", 'RED')
            print(response.text)
            return
        
        result = response.json()
        session_id = result['session_id']
        print_colored(f"✅ Complex workflow started! Session: {session_id}", 'GREEN')
        print_colored(f"🤖 {result['current_agent_name']} Analysis:", 'PURPLE')
        print(result['response'][:500] + "..." if len(result['response']) > 500 else result['response'])
        
        # Wait for first agent and move through others with proper completion checking
        for i in range(len(complex_agents) - 1):
            # Wait for current agent to complete
            if not wait_for_agent_completion(session_id, i):
                print_colored(f"❌ Agent {i} did not complete in time", 'RED')
                break
                
            print_colored(f"\n➡️ Moving to {complex_agents[i+1]['name']}...", 'YELLOW')
            
            next_response = requests.post(f"{BASE_URL}/api/v1/agentic-ai/next?session_id={session_id}", timeout=120)
            
            if next_response.status_code == 200:
                next_result = next_response.json()
                print_colored(f"🤖 {next_result['current_agent_name']}:", 'PURPLE')
                print(next_result['response'][:500] + "..." if len(next_result['response']) > 500 else next_result['response'])
            else:
                print_colored(f"❌ Failed at agent {i+1}: {next_response.status_code}", 'RED')
                print(next_response.text)
                break
        
        # Wait for final agent to complete
        if not wait_for_agent_completion(session_id, len(complex_agents) - 1):
            print_colored("❌ Final agent did not complete", 'RED')
            return
        
        # Get final status
        print_colored("\n📊 Getting final workflow status...", 'BLUE')
        status_response = requests.get(f"{BASE_URL}/api/v1/agentic-ai/status/{session_id}", timeout=30)
        
        if status_response.status_code == 200:
            status = status_response.json()
            print_colored(f"✅ Workflow Complete: {status['is_complete']}", 'GREEN')
            print_colored(f"Total Agents: {status['total_agents']}", 'CYAN')
            print_colored(f"Agents Completed: {len(status['agent_responses'])}", 'CYAN')
            
            completion_status = status.get('completion_status', {})
            for agent_key, completed in completion_status.items():
                status_icon = "✅" if completed else "❌"
                print_colored(f"  {status_icon} {agent_key}: {'Completed' if completed else 'Not completed'}", 'CYAN')
            
        else:
            print_colored(f"❌ Failed to get final status: {status_response.status_code}", 'RED')
            
    except Exception as e:
        print_colored(f"❌ Complex workflow failed: {e}", 'RED')

def main():

    
    try:

        test_feedback_mechanism()


        
    except KeyboardInterrupt:
        print_colored("\n\n⚠️  Demo interrupted by user", 'YELLOW')
    except Exception as e:
        print_colored(f"\n❌ Demo failed: {e}", 'RED')


def test_feedback_mechanism():
    """Test the feedback mechanism with a proper single-agent workflow"""
    print_header("Testing Feedback Mechanism")
    
    # Create a single-agent workflow for feedback testing
    request_data = {
        "agents": [
            {
                "name": "Math_Calculator",
                "persona": "You are a helpful calculator who solves math problems step by step with clear explanations.",
                "task": "Solve mathematical problems with detailed explanations, showing all work and reasoning."
            },
            {
                "name": "Concept_Explainer", 
                "persona": "You are an educational specialist who explains mathematical concepts clearly.",
                "task": "Explain the underlying concepts and provide additional examples."
            },
            {
                "name": "Practice_Generator",
                "persona": "You are a practice problem generator who creates similar problems for reinforcement.", 
                "task": "Generate practice problems and solutions based on the previous work."
            }
        ],
        "initial_task": "Calculate 15% of 240 and explain how to calculate percentages in general."
    }
    
    try:
        print_colored("📤 Starting new workflow for feedback test...", 'BLUE')
        response = requests.post(f"{BASE_URL}/api/v1/agentic-ai", json=request_data, timeout=60)
        
        if response.status_code != 200:
            print_colored(f"❌ Failed to start feedback test workflow: {response.status_code}", 'RED')
            print(response.text)
            return
            
        result = response.json()
        session_id = result['session_id']
        
        print_colored(f"✅ Initial response from {result['current_agent_name']}:", 'GREEN')
        print(result['response'][:300] + "..." if len(result['response']) > 300 else result['response'])
        
        # Wait for agent to complete
        if not wait_for_agent_completion(session_id, 0):
            return
            
        # Feedback loop for Agent 1
        agent1_satisfied = False
        while not agent1_satisfied:
            print_colored("\n💬 Agent 1 (Math Calculator) Options:", 'CYAN')
            print("1. Provide feedback to improve the response")
            print("2. Satisfied - Continue to next agent")
            
            choice = input("\nEnter your choice (1-2): ").strip()
            
            if choice == "1":
                # Get feedback from user
                print_colored("\n📝 Enter your feedback for the Math Calculator:", 'YELLOW')
                print("(Example: 'Can you also show the decimal method and explain when to use different approaches?')")
                user_feedback = input("Your feedback: ").strip()
                
                if user_feedback:
                    feedback_data = {
                        "session_id": session_id,
                        "agent_index": 0,
                        "feedback": user_feedback
                    }
                    
                    print_colored("\n🔄 Processing your feedback...", 'YELLOW')
                    feedback_response = requests.post(f"{BASE_URL}/api/v1/agentic-ai/feedback", json=feedback_data, timeout=60)
                    
                    if feedback_response.status_code == 200:
                        feedback_result = feedback_response.json()
                        print_colored("✅ Feedback processed! Updated response:", 'GREEN')
                        print(feedback_result['response'][:400] + "..." if len(feedback_result['response']) > 400 else feedback_result['response'])
                        # Continue the loop to ask if they want more feedback or are satisfied
                    else:
                        print_colored(f"❌ Feedback failed: {feedback_response.status_code}", 'RED')
                        print(feedback_response.text)
                        # Continue the loop to try again
                else:
                    print_colored("⚠️ No feedback provided.", 'YELLOW')
                    # Continue the loop to ask again
                    
            elif choice == "2":
                print_colored("✅ Agent 1 completed. Moving to next agent.", 'GREEN')
                agent1_satisfied = True
            else:
                print_colored("❌ Invalid choice. Please enter 1 or 2.", 'RED')
        
        # Continue to next agent
        print_colored("\n➡️ Moving to next agent (Concept Explainer)...", 'YELLOW')
        next_response = requests.post(f"{BASE_URL}/api/v1/agentic-ai/next?session_id={session_id}", timeout=60)
        
        if next_response.status_code == 200:
            next_result = next_response.json()
            print_colored(f"🤖 {next_result['current_agent_name']} response:", 'PURPLE')
            print(next_result['response'][:400] + "..." if len(next_result['response']) > 400 else next_result['response'])
            
            # Wait for agent 2 to complete
            if not wait_for_agent_completion(session_id, 1):
                return
            
            # Feedback loop for Agent 2
            agent2_satisfied = False
            while not agent2_satisfied:
                print_colored("\n💬 Agent 2 (Concept Explainer) Options:", 'CYAN')
                print("1. Provide feedback to improve the response")
                print("2. Satisfied - Continue to next agent")
                
                choice = input("\nEnter your choice (1-2): ").strip()
                
                if choice == "1":
                    # Get feedback from user
                    print_colored("\n📝 Enter your feedback for the Concept Explainer:", 'YELLOW')
                    print("(Example: 'Can you provide more real-world examples of when percentages are used?')")
                    user_feedback = input("Your feedback: ").strip()
                    
                    if user_feedback:
                        feedback_data = {
                            "session_id": session_id,
                            "agent_index": 1,
                            "feedback": user_feedback
                        }
                        
                        print_colored("\n🔄 Processing your feedback...", 'YELLOW')
                        feedback_response = requests.post(f"{BASE_URL}/api/v1/agentic-ai/feedback", json=feedback_data, timeout=60)
                        
                        if feedback_response.status_code == 200:
                            feedback_result = feedback_response.json()
                            print_colored("✅ Feedback processed! Updated response:", 'GREEN')
                            print(feedback_result['response'][:400] + "..." if len(feedback_result['response']) > 400 else feedback_result['response'])
                            # Continue the loop to ask if they want more feedback or are satisfied
                        else:
                            print_colored(f"❌ Feedback failed: {feedback_response.status_code}", 'RED')
                            print(feedback_response.text)
                            # Continue the loop to try again
                    else:
                        print_colored("⚠️ No feedback provided.", 'YELLOW')
                        # Continue the loop to ask again
                        
                elif choice == "2":
                    print_colored("✅ Agent 2 completed. Feedback test finished!", 'GREEN')
                    agent2_satisfied = True
                else:
                    print_colored("❌ Invalid choice. Please enter 1 or 2.", 'RED')
        else:
            print_colored(f"❌ Failed to move to next agent: {next_response.status_code}", 'RED')
            
    except Exception as e:
        print_colored(f"❌ Feedback test failed: {e}", 'RED')

if __name__ == "__main__":
    main()

"""
Example Agent Configurations for Different Use Cases
This file contains pre-built agent configurations for common workflows
"""

# Complex Mathematical Problem Solving Workflow
MATH_OPTIMIZATION_WORKFLOW = {
    "agents": [
        {
            "name": "Mathematical_Analyst",
            "persona": "You are a senior mathematical analyst specializing in optimization problems, calculus, and statistical analysis.",
            "task": "Analyze the mathematical problem, identify key variables, constraints, and objectives. Set up the mathematical framework for solution."
        },
        {
            "name": "Optimization_Specialist", 
            "persona": "You are an operations research expert skilled in linear programming, nonlinear optimization, and constraint satisfaction.",
            "task": "Using the mathematical framework, apply optimization techniques to find optimal solutions. Show detailed calculations and methodology."
        },
        {
            "name": "Results_Validator",
            "persona": "You are a verification specialist who validates mathematical solutions and performs sensitivity analysis.",
            "task": "Verify the optimization results, check for edge cases, and perform sensitivity analysis on key parameters."
        },
        {
            "name": "Business_Interpreter",
            "persona": "You are a business analyst who translates technical mathematical results into actionable business insights.",
            "task": "Interpret the mathematical results in business terms and provide clear recommendations and implementation strategies."
        }
    ],
    "description": "Solves complex business optimization problems requiring mathematical analysis"
}

# Content Creation Workflow
CONTENT_CREATION_WORKFLOW = {
    "agents": [
        {
            "name": "Research_Specialist",
            "persona": "You are a thorough research specialist who gathers comprehensive information on any topic using systematic research methods.",
            "task": "Research the topic extensively, gather relevant facts, statistics, and current trends. Provide a solid foundation of information."
        },
        {
            "name": "Content_Strategist",
            "persona": "You are a content strategy expert who plans engaging content structure, tone, and audience targeting.",
            "task": "Based on the research, develop a content strategy including structure, key messages, target audience considerations, and engagement tactics."
        },
        {
            "name": "Creative_Writer",
            "persona": "You are a skilled creative writer who crafts compelling, engaging content that resonates with audiences.",
            "task": "Write the actual content following the strategy, ensuring it's engaging, well-structured, and compelling for the target audience."
        },
        {
            "name": "Editor_Reviewer",
            "persona": "You are a professional editor who polishes content for clarity, impact, and error-free presentation.",
            "task": "Review and edit the content for grammar, clarity, flow, and impact. Ensure the final piece meets professional standards."
        }
    ],
    "description": "Creates high-quality content from research to final polished piece"
}

# Software Development Code Review Workflow
SOFTWARE_REVIEW_WORKFLOW = {
    "agents": [
        {
            "name": "Code_Analyzer",
            "persona": "You are a senior software engineer who analyzes code architecture, design patterns, and technical implementation.",
            "task": "Analyze the code structure, design patterns, architecture decisions, and technical implementation. Identify strengths and areas for improvement."
        },
        {
            "name": "Security_Auditor",
            "persona": "You are a cybersecurity expert specializing in secure coding practices and vulnerability assessment.",
            "task": "Review the code for security vulnerabilities, potential attack vectors, and compliance with security best practices."
        },
        {
            "name": "Performance_Optimizer",
            "persona": "You are a performance engineering specialist who identifies bottlenecks and optimization opportunities.",
            "task": "Analyze the code for performance issues, suggest optimizations, and recommend improvements for scalability and efficiency."
        }
    ],
    "description": "Comprehensive code review covering architecture, security, and performance"
}

# Business Strategy Development Workflow
BUSINESS_STRATEGY_WORKFLOW = {
    "agents": [
        {
            "name": "Market_Analyst",
            "persona": "You are a market research analyst who specializes in competitive analysis and market trends identification.",
            "task": "Analyze the market landscape, competition, trends, and opportunities. Provide comprehensive market intelligence."
        },
        {
            "name": "Financial_Planner",
            "persona": "You are a financial strategist who develops financial models, projections, and investment strategies.",
            "task": "Create financial projections, analyze costs and revenues, and develop financial strategies based on market analysis."
        },
        {
            "name": "Operations_Consultant",
            "persona": "You are an operations consultant who designs efficient processes and implementation strategies.",
            "task": "Design operational strategies, process improvements, and implementation plans based on market and financial analysis."
        },
        {
            "name": "Strategic_Advisor",
            "persona": "You are a senior business strategist who synthesizes analysis into actionable strategic recommendations.",
            "task": "Synthesize all analyses into a comprehensive strategic plan with clear recommendations and action items."
        }
    ],
    "description": "Develops comprehensive business strategy from market analysis to implementation plan"
}

# Academic Research Workflow
ACADEMIC_RESEARCH_WORKFLOW = {
    "agents": [
        {
            "name": "Literature_Reviewer",
            "persona": "You are an academic researcher skilled in systematic literature reviews and scholarly research methods.",
            "task": "Conduct a comprehensive literature review, identify key studies, theories, and gaps in current knowledge."
        },
        {
            "name": "Methodology_Designer",
            "persona": "You are a research methodology expert who designs rigorous research approaches and experimental designs.",
            "task": "Design appropriate research methodology, including study design, data collection methods, and analytical approaches."
        },
        {
            "name": "Data_Analyst",
            "persona": "You are a quantitative analyst specializing in statistical analysis and data interpretation.",
            "task": "Analyze data using appropriate statistical methods and interpret results in the context of existing research."
        },
        {
            "name": "Academic_Writer",
            "persona": "You are an academic writer who crafts scholarly papers following academic standards and conventions.",
            "task": "Write the research findings in proper academic format, ensuring clarity, rigor, and adherence to scholarly standards."
        }
    ],
    "description": "Conducts academic research from literature review to final paper"
}

# Simple Math Tutoring Workflow
MATH_TUTORING_WORKFLOW = {
    "agents": [
        {
            "name": "Problem_Analyzer",
            "persona": "You are a patient math tutor who breaks down complex problems into manageable steps.",
            "task": "Analyze the math problem, identify the key concepts involved, and break it down into clear, logical steps."
        },
        {
            "name": "Solution_Developer",
            "persona": "You are an experienced mathematics teacher who demonstrates problem-solving techniques step by step.",
            "task": "Solve the problem step by step, showing all work and explaining the reasoning behind each step."
        },
        {
            "name": "Concept_Explainer",
            "persona": "You are an educational specialist who explains mathematical concepts in an easy-to-understand way.",
            "task": "Explain the underlying mathematical concepts and provide additional examples to reinforce understanding."
        }
    ],
    "description": "Provides comprehensive math tutoring from problem analysis to concept reinforcement"
}

# Product Design Workflow
PRODUCT_DESIGN_WORKFLOW = {
    "agents": [
        {
            "name": "User_Researcher",
            "persona": "You are a UX researcher who understands user needs, behaviors, and pain points through systematic research.",
            "task": "Research user needs, analyze user behavior patterns, and identify key pain points and opportunities."
        },
        {
            "name": "Product_Designer",
            "persona": "You are a product designer who creates user-centered solutions and intuitive interfaces.",
            "task": "Design product solutions based on user research, create user flows, and define key features and functionality."
        },
        {
            "name": "Technical_Architect",
            "persona": "You are a technical architect who evaluates feasibility and designs system architecture.",
            "task": "Assess technical feasibility, design system architecture, and identify technical requirements and constraints."
        },
        {
            "name": "Product_Manager",
            "persona": "You are a product manager who creates roadmaps, prioritizes features, and plans product development.",
            "task": "Create product roadmap, prioritize features, and develop implementation strategy based on user needs and technical constraints."
        }
    ],
    "description": "Designs products from user research to development roadmap"
}

# Example Usage Functions
def get_workflow_config(workflow_type: str) -> dict:
    """Get a workflow configuration by type"""
    workflows = {
        "math_optimization": MATH_OPTIMIZATION_WORKFLOW,
        "content_creation": CONTENT_CREATION_WORKFLOW,
        "software_review": SOFTWARE_REVIEW_WORKFLOW,
        "business_strategy": BUSINESS_STRATEGY_WORKFLOW,
        "academic_research": ACADEMIC_RESEARCH_WORKFLOW,
        "math_tutoring": MATH_TUTORING_WORKFLOW,
        "product_design": PRODUCT_DESIGN_WORKFLOW
    }
    
    return workflows.get(workflow_type, {})

def list_available_workflows() -> list:
    """List all available workflow types"""
    return [
        "math_optimization",
        "content_creation", 
        "software_review",
        "business_strategy",
        "academic_research",
        "math_tutoring",
        "product_design"
    ]

def create_custom_workflow(agents_data: list, description: str = "") -> dict:
    """Create a custom workflow configuration"""
    return {
        "agents": agents_data,
        "description": description
    }

# Test Functions for Each Workflow Type
def test_math_optimization():
    """Test the math optimization workflow with a complex problem"""
    return {
        "workflow": MATH_OPTIMIZATION_WORKFLOW,
        "test_problem": """
        A tech startup wants to optimize their resource allocation for maximum growth:
        
        Resources:
        - Engineering team: 20 developers (cost: $150k/year each)
        - Marketing budget: $500k/year
        - Fixed costs: $200k/year
        - Revenue per customer: $100/month
        - Customer acquisition cost: $50 per customer
        
        Constraints:
        - Cannot hire more than 30 developers total
        - Marketing budget cannot exceed $800k/year
        - Must maintain at least 6 months runway
        
        Find the optimal allocation to maximize revenue growth over 2 years.
        """
    }

def test_content_creation():
    """Test the content creation workflow"""
    return {
        "workflow": CONTENT_CREATION_WORKFLOW,
        "test_problem": "Create a comprehensive blog post about the future of artificial intelligence in healthcare, targeting healthcare professionals and technology decision-makers."
    }

def test_software_review():
    """Test the software review workflow"""
    return {
        "workflow": SOFTWARE_REVIEW_WORKFLOW,
        "test_problem": """
        Review this Python web scraping function:
        
        ```python
        import requests
        from bs4 import BeautifulSoup
        
        def scrape_data(url):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            data = []
            for item in soup.find_all('div', class_='item'):
                title = item.find('h2').text
                price = item.find('span', class_='price').text
                data.append({'title': title, 'price': price})
            return data
        ```
        """
    }

# Example Integration with FastAPI Endpoint
EXAMPLE_REQUESTS = {
    "complex_math": {
        "agents": MATH_OPTIMIZATION_WORKFLOW["agents"],
        "initial_task": test_math_optimization()["test_problem"]
    },
    "content_creation": {
        "agents": CONTENT_CREATION_WORKFLOW["agents"], 
        "initial_task": test_content_creation()["test_problem"]
    },
    "code_review": {
        "agents": SOFTWARE_REVIEW_WORKFLOW["agents"],
        "initial_task": test_software_review()["test_problem"]
    }
}

if __name__ == "__main__":
    # Example usage
    print("Available Workflows:")
    for workflow_type in list_available_workflows():
        config = get_workflow_config(workflow_type)
        print(f"- {workflow_type}: {config.get('description', 'No description')}")
        print(f"  Agents: {len(config.get('agents', []))}")
        print()
    
    # Test a specific workflow
    math_test = test_math_optimization()
    print("Math Optimization Test Problem:")
    print(math_test["test_problem"])
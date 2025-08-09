import json
import hashlib
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import importlib.util
import sys
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class LLMProcessor:
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize LLM processor with configuration from config file.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.llm_config = config.get('llm', {})
        
        # Environment variable substitution for API key
        api_key = self.llm_config.get('api_key', '')
        if api_key.startswith('${') and api_key.endswith('}'):
            env_var = api_key[2:-1]
            api_key = os.getenv(env_var)
            if not api_key:
                raise ValueError(f"Environment variable {env_var} not found")
        
        self.provider = self.llm_config.get('provider', 'openai')
        self.model = self.llm_config.get('model', 'gpt-4o-mini')
        self.temperature = self.llm_config.get('temperature', 0.1)
        self.max_tokens = self.llm_config.get('max_tokens', 1000)
        base_url = self.llm_config.get('base_url')
        
        # Initialize OpenAI client (works for OpenAI-compatible endpoints too)
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        
        logger.info(f"Initialized LLM processor with provider: {self.provider}, model: {self.model}, base_url: {base_url}")
        
    def generate_prompt_hash(self, prompt: str) -> str:
        """Generate hash for prompt to enable caching."""
        return hashlib.sha256(prompt.encode()).hexdigest()
    
    def load_prompt_function(self, prompt_file: str) -> callable:
        """
        Dynamically load prompt rendering function from file.
        
        Args:
            prompt_file: Path to prompt file (e.g., 'summary_prompt.py')
            
        Returns:
            Callable prompt rendering function
        """
        prompt_path = Path("prompt-lib") / prompt_file
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        # Load module dynamically
        spec = importlib.util.spec_from_file_location("prompt_module", prompt_path)
        prompt_module = importlib.util.module_from_spec(spec)
        sys.modules["prompt_module"] = prompt_module
        spec.loader.exec_module(prompt_module)
        
        # Extract the render function (assumes naming convention)
        function_name = f"render_{prompt_file.split('.')[0]}"
        
        if not hasattr(prompt_module, function_name):
            raise AttributeError(f"Function {function_name} not found in {prompt_file}")
        
        return getattr(prompt_module, function_name)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(
            retry_if_exception_type(openai.RateLimitError) |
            retry_if_exception_type(openai.APIConnectionError) |
            retry_if_exception_type(openai.APITimeoutError) |
            retry_if_exception_type(openai.InternalServerError)
        ),
    )
    def call_llm(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """
        Make API call to LLM with retry mechanism.
        
        Args:
            prompt: The prompt to send
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            str: LLM response
        """
        try:
            # Use instance defaults if not provided
            if temperature is None:
                temperature = self.temperature
            if max_tokens is None:
                max_tokens = self.max_tokens
                
            logger.debug(f"Making LLM API call with provider: {self.provider}, model: {self.model}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result = response.choices[0].message.content
            logger.debug(f"LLM API call successful, response length: {len(result)}")
            
            return result
            
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit hit: {str(e)}")
            raise
        except openai.APIConnectionError as e:
            logger.warning(f"API connection error: {str(e)}")
            raise
        except openai.APITimeoutError as e:
            logger.warning(f"API timeout: {str(e)}")
            raise
        except openai.InternalServerError as e:
            logger.warning(f"OpenAI internal server error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LLM call: {str(e)}")
            raise
    
    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response from LLM, handling common formatting issues.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Dict: Parsed JSON response
        """
        try:
            # Try to parse directly first
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Try to find JSON-like content
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            
            logger.error(f"Failed to parse JSON response: {response[:500]}...")
            raise ValueError("Could not parse JSON from LLM response")
    
    def process_single_row(self, row: Dict[str, Any], task_name: str, data_case: str, 
                          prompt_file: str, db_manager, use_cache: bool = True) -> Dict[str, Any]:
        """
        Process a single row of data through the LLM pipeline.
        
        Args:
            row: Data row to process
            task_name: Name of the task
            data_case: Data case name
            prompt_file: Prompt file to use
            db_manager: Database manager instance for caching
            use_cache: Whether to use cached results
            
        Returns:
            Dict: Processing result with metadata
        """
        try:
            # Load prompt function
            render_prompt = self.load_prompt_function(prompt_file)
            
            # Generate prompt
            prompt = render_prompt(data_case, row)
            prompt_hash = self.generate_prompt_hash(prompt)
            
            # Check cache first
            if use_cache:
                cached_result = db_manager.get_cached_llm_result(prompt_hash)
                if cached_result:
                    logger.debug(f"Using cached result for prompt hash: {prompt_hash}")
                    return {
                        'result': cached_result,
                        'cached': True,
                        'processing_time': 0,
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Process with LLM
            start_time = datetime.now()
            
            llm_response = self.call_llm(prompt)
            parsed_result = self.parse_json_response(llm_response)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Cache the result
            if use_cache:
                db_manager.cache_llm_result(prompt_hash, parsed_result)
            
            result = {
                'result': parsed_result,
                'cached': False,
                'processing_time': processing_time,
                'timestamp': end_time.isoformat(),
                'prompt_hash': prompt_hash
            }
            
            logger.info(f"Successfully processed row in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process row: {str(e)}")
            raise
    
    def validate_response_schema(self, response: Dict[str, Any], expected_fields: List[str]) -> bool:
        """
        Validate that the response contains expected fields.
        
        Args:
            response: Parsed response from LLM
            expected_fields: List of required fields
            
        Returns:
            bool: True if valid
        """
        missing_fields = [field for field in expected_fields if field not in response]
        
        if missing_fields:
            logger.warning(f"Response missing required fields: {missing_fields}")
            return False
        
        return True
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the current model and configuration."""
        return {
            'provider': self.provider,
            'model': self.model,
            'base_url': self.llm_config.get('base_url'),
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
import asyncio
import hashlib
import json
import logging
import pandas as pd
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import traceback

import diskcache as dc
from pydantic import BaseModel, ValidationError
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.outputs import RequestOutput


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vllm_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for the batch processor"""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_num_seqs: int = 64
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85
    max_tokens: int = 3000
    temperature: float = 0.8
    top_p: float = 0.95
    cache_dir: str = "./llm_cache"
    batch_size: int = 32
    max_retries: int = 3
    retry_delay: float = 1.0
    checkpoint_interval: int = 100


class ProcessingStats:
    """Track processing statistics"""
    def __init__(self):
        self.total_requests = 0
        self.completed = 0
        self.cache_hits = 0
        self.errors = 0
        self.start_time = time.time()
        
    def update_completed(self, from_cache: bool = False):
        self.completed += 1
        if from_cache:
            self.cache_hits += 1
            
    def update_error(self):
        self.errors += 1
        
    def get_progress_stats(self) -> Dict:
        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0
        
        return {
            "total": self.total_requests,
            "completed": self.completed,
            "cache_hits": self.cache_hits,
            "errors": self.errors,
            "progress": f"{self.completed}/{self.total_requests} ({self.completed/self.total_requests*100:.1f}%)",
            "rate": f"{rate:.2f} req/s",
            "elapsed": f"{elapsed:.1f}s"
        }


class CacheManager:
    """Manages disk-based caching with hashing"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache = dc.Cache(str(self.cache_dir))
        
    def _generate_key(self, prompt: str, schema_dict: Dict, config: ProcessingConfig) -> str:
        """Generate a hash key for caching"""
        # Create a deterministic string from all inputs that affect output
        key_data = {
            "prompt": prompt,
            "schema": schema_dict,
            "model": config.model_name,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(self, prompt: str, schema_dict: Dict, config: ProcessingConfig) -> Optional[Dict]:
        """Get cached result if it exists"""
        key = self._generate_key(prompt, schema_dict, config)
        return self.cache.get(key)
    
    def set(self, prompt: str, schema_dict: Dict, config: ProcessingConfig, result: Dict):
        """Cache the result"""
        key = self._generate_key(prompt, schema_dict, config)
        self.cache.set(key, result)
    
    def close(self):
        """Close the cache"""
        self.cache.close()


class PromptRenderer:
    """Efficiently renders prompts with data substitution"""
    
    @staticmethod
    def render_prompt(template: str, row_data: Dict) -> str:
        """Render a prompt template with row data"""
        try:
            # Support both f-string style {column} and format() style substitution
            return template.format(**row_data)
        except KeyError as e:
            raise ValueError(f"Template references missing column: {e}")
    
    @staticmethod
    def pre_render_all_prompts(template: str, df: pd.DataFrame) -> List[str]:
        """Pre-render all prompts to avoid repeated rendering"""
        logger.info(f"Pre-rendering {len(df)} prompts...")
        prompts = []
        
        for idx, row in df.iterrows():
            try:
                row_dict = row.to_dict()
                prompt = PromptRenderer.render_prompt(template, row_dict)
                prompts.append(prompt)
            except Exception as e:
                logger.error(f"Error rendering prompt for row {idx}: {e}")
                prompts.append(None)  # Mark as failed
                
        valid_prompts = [p for p in prompts if p is not None]
        logger.info(f"Successfully rendered {len(valid_prompts)}/{len(prompts)} prompts")
        return prompts


class AsyncVLLMProcessor:
    """Main async processor for vLLM with caching and error handling"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.cache_manager = CacheManager(config.cache_dir)
        self.stats = ProcessingStats()
        self.engine: Optional[AsyncLLMEngine] = None
        self.shutdown_requested = False
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    async def initialize_engine(self):
        """Initialize the vLLM async engine"""
        try:
            engine_args = AsyncEngineArgs(
                model=self.config.model_name,
                max_num_seqs=self.config.max_num_seqs,
                max_model_len=self.config.max_model_len,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                dtype="bfloat16"
            )
            
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info(f"Successfully initialized vLLM engine with model: {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            raise
    
    async def process_single_request(
        self, 
        prompt: str, 
        schema: BaseModel, 
        request_id: str,
        max_retries: int = None
    ) -> Optional[Dict]:
        """Process a single request with retries and error handling"""
        
        if max_retries is None:
            max_retries = self.config.max_retries
            
        # Convert schema to dict for caching
        schema_dict = schema.model_json_schema()
        
        # Check cache first
        cached_result = self.cache_manager.get(prompt, schema_dict, self.config)
        if cached_result is not None:
            self.stats.update_completed(from_cache=True)
            return cached_result
        
        # Process with vLLM
        for attempt in range(max_retries + 1):
            try:
                if self.shutdown_requested:
                    logger.info("Shutdown requested, stopping processing")
                    return None
                
                sampling_params = SamplingParams(
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_tokens,
                    guided_json=schema_dict  # For structured output
                )
                
                # Generate with vLLM
                async for output in self.engine.generate(
                    prompt, 
                    sampling_params, 
                    request_id=request_id
                ):
                    if output.finished:
                        # Parse and validate the JSON output
                        generated_text = output.outputs[0].text.strip()
                        
                        try:
                            result_json = json.loads(generated_text)
                            
                            # Validate against Pydantic schema
                            validated_result = schema(**result_json)
                            final_result = validated_result.model_dump()
                            
                            # Cache the result
                            self.cache_manager.set(prompt, schema_dict, self.config, final_result)
                            self.stats.update_completed()
                            
                            return final_result
                            
                        except (json.JSONDecodeError, ValidationError) as e:
                            logger.warning(f"Invalid JSON/schema for request {request_id}: {e}")
                            if attempt < max_retries:
                                await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                                continue
                            else:
                                raise
                                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for request {request_id}: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    self.stats.update_error()
                    logger.error(f"All attempts failed for request {request_id}: {e}")
                    return None
        
        return None
    
    async def process_batch(
        self, 
        prompts: List[str], 
        schema: BaseModel,
        start_index: int = 0
    ) -> List[Optional[Dict]]:
        """Process a batch of prompts with concurrency control"""
        
        # Filter out None prompts (failed renders)
        valid_prompts = [(i, p) for i, p in enumerate(prompts) if p is not None]
        
        self.stats.total_requests = len(valid_prompts)
        logger.info(f"Processing {len(valid_prompts)} valid prompts in batches of {self.config.batch_size}")
        
        results = [None] * len(prompts)  # Initialize with None for all indices
        
        # Process in batches to control concurrency
        for batch_start in range(0, len(valid_prompts), self.config.batch_size):
            if self.shutdown_requested:
                logger.info("Shutdown requested, stopping batch processing")
                break
                
            batch_end = min(batch_start + self.config.batch_size, len(valid_prompts))
            batch_items = valid_prompts[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//self.config.batch_size + 1}: items {batch_start}-{batch_end-1}")
            
            # Create tasks for this batch
            tasks = []
            for original_idx, prompt in batch_items:
                request_id = f"req_{start_index + original_idx}"
                task = self.process_single_request(prompt, schema, request_id)
                tasks.append((original_idx, task))
            
            # Wait for batch completion
            for original_idx, task in tasks:
                try:
                    result = await task
                    results[original_idx] = result
                except Exception as e:
                    logger.error(f"Batch task failed for index {original_idx}: {e}")
                    results[original_idx] = None
            
            # Log progress
            stats = self.stats.get_progress_stats()
            logger.info(f"Progress: {stats['progress']} | Rate: {stats['rate']} | Cache hits: {self.stats.cache_hits}")
            
            # Checkpoint periodically
            if (batch_start + self.config.batch_size) % (self.config.checkpoint_interval * self.config.batch_size) == 0:
                logger.info("Checkpoint: Progress saved to cache")
        
        return results
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.engine:
                # vLLM doesn't have explicit cleanup, but we can log
                logger.info("Cleaning up vLLM engine...")
            
            self.cache_manager.close()
            logger.info("Cache manager closed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Example usage functions
def create_example_schema():
    """Example Pydantic schema for structured output"""
    class CodeResponse(BaseModel):
        language: str
        code: str
        explanation: str
        complexity: str
        
    return CodeResponse


async def main():
    """Example main function demonstrating usage"""
    
    # Configuration
    config = ProcessingConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        max_num_seqs=64,
        batch_size=32,
        cache_dir="./llm_cache",
        max_retries=3
    )
    
    # Example data
    df = pd.DataFrame({
        'language': ['Python', 'JavaScript', 'Rust', 'Go', 'Java'],
        'task': ['fibonacci', 'hello world', 'web server', 'file reader', 'sorting algorithm'],
        'difficulty': ['medium', 'easy', 'hard', 'medium', 'easy']
    })
    
    # Prompt template
    prompt_template = """
    Write a {task} program in {language}. The difficulty level should be {difficulty}.
    
    Please provide:
    1. The complete code
    2. A brief explanation of how it works
    3. The complexity level
    
    Format your response as JSON with the following structure:
    - language: the programming language used
    - code: the complete code
    - explanation: brief explanation
    - complexity: complexity assessment
    """
    
    # Pre-render prompts
    rendered_prompts = PromptRenderer.pre_render_all_prompts(prompt_template, df)
    
    # Create schema
    schema = create_example_schema()
    
    # Initialize processor
    processor = AsyncVLLMProcessor(config)
    
    try:
        # Initialize engine
        await processor.initialize_engine()
        
        # Process all prompts
        results = await processor.process_batch(rendered_prompts, schema)
        
        # Save results
        output_df = df.copy()
        output_df['llm_response'] = results
        output_df.to_json('results.json', orient='records', indent=2)
        
        # Final stats
        final_stats = processor.stats.get_progress_stats()
        logger.info(f"Processing complete! Final stats: {final_stats}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        logger.error(traceback.format_exc())
        
    finally:
        await processor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())



#!/usr/bin/env python3
"""
Example usage of the Async vLLM Processor with real CSV data
"""

import asyncio
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional

# Import from the main processor (assuming it's saved as vllm_processor.py)
from vllm_processor import AsyncVLLMProcessor, ProcessingConfig, PromptRenderer


# Define your Pydantic schema for structured output
class ProductAnalysis(BaseModel):
    """Schema for product analysis output"""
    product_name: str = Field(description="Name of the product")
    category: str = Field(description="Product category")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    key_features: List[str] = Field(description="List of key features mentioned")
    rating_prediction: float = Field(description="Predicted rating from 1.0 to 5.0", ge=1.0, le=5.0)
    summary: str = Field(description="Brief summary of the analysis")
    recommendations: List[str] = Field(description="List of recommendations for improvement")


class ReviewAnalysis(BaseModel):
    """Schema for review analysis"""
    sentiment: str = Field(description="positive, negative, or neutral")
    emotion: str = Field(description="Primary emotion detected")
    rating_prediction: int = Field(description="Predicted rating 1-5", ge=1, le=5)
    key_points: List[str] = Field(description="Main points from the review")
    summary: str = Field(description="One sentence summary")


async def process_product_reviews():
    """Example: Process product reviews from CSV"""
    
    # Create sample data
    reviews_data = {
        'product_id': [1, 2, 3, 4, 5],
        'product_name': [
            'Wireless Headphones Pro', 
            'Smart Coffee Maker', 
            'Gaming Keyboard RGB',
            'Fitness Tracker Watch',
            'Bluetooth Speaker Mini'
        ],
        'review_text': [
            'Amazing sound quality and battery life lasts all day. Comfortable fit.',
            'Makes perfect coffee every time. Easy to program and clean.',
            'Keys feel great and lighting is customizable. Some keys stick occasionally.',
            'Accurate fitness tracking but battery drains faster than expected.',
            'Compact size but sound is tinny. Good for the price though.'
        ],
        'category': ['Electronics', 'Appliances', 'Electronics', 'Wearables', 'Electronics'],
        'price': [199.99, 89.99, 129.99, 149.99, 39.99]
    }
    
    df = pd.DataFrame(reviews_data)
    
    # Save to CSV for demonstration
    df.to_csv('sample_reviews.csv', index=False)
    print(f"Created sample CSV with {len(df)} reviews")
    
    # Define prompt template
    prompt_template = """
    Analyze the following product review:
    
    Product: {product_name}
    Category: {category}
    Price: ${price}
    Review: "{review_text}"
    
    Please analyze this review and provide:
    1. Overall sentiment (positive, negative, or neutral)
    2. Primary emotion detected
    3. Predicted rating (1-5 scale)
    4. Key points mentioned in the review
    5. A brief one-sentence summary
    
    Consider the product category and price point in your analysis.
    """
    
    # Configuration for processing
    config = ProcessingConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",  # Use a model you have access to
        max_num_seqs=32,
        batch_size=16,
        cache_dir="./review_cache",
        max_retries=2,
        max_tokens=1000,
        temperature=0.3  # Lower temperature for more consistent analysis
    )
    
    # Pre-render all prompts
    rendered_prompts = PromptRenderer.pre_render_all_prompts(prompt_template, df)
    
    # Initialize processor
    processor = AsyncVLLMProcessor(config)
    
    try:
        print("Initializing vLLM engine...")
        await processor.initialize_engine()
        
        print("Processing reviews...")
        results = await processor.process_batch(rendered_prompts, ReviewAnalysis)
        
        # Combine results with original data
        df['analysis'] = results
        
        # Save results
        output_file = 'review_analysis_results.json'
        df.to_json(output_file, orient='records', indent=2)
        print(f"Results saved to {output_file}")
        
        # Print summary
        successful_results = [r for r in results if r is not None]
        print(f"\nProcessing Summary:")
        print(f"Total reviews: {len(df)}")
        print(f"Successfully processed: {len(successful_results)}")
        print(f"Cache hits: {processor.stats.cache_hits}")
        print(f"Errors: {processor.stats.errors}")
        
        # Show sample results
        if successful_results:
            print(f"\nSample Analysis:")
            for i, (_, row) in enumerate(df.iterrows()):
                if row['analysis'] is not None:
                    print(f"\nProduct: {row['product_name']}")
                    print(f"Original Review: {row['review_text'][:100]}...")
                    print(f"Analysis: {row['analysis']}")
                    break
                    
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await processor.cleanup()


async def process_large_dataset():
    """Example: Process a large dataset with crash recovery"""
    
    # Simulate large dataset
    large_data = []
    for i in range(1000):  # 1000 items for demonstration
        large_data.append({
            'id': i,
            'customer_name': f'Customer_{i}',
            'feedback': f'This is customer feedback number {i}. The service was {"excellent" if i % 3 == 0 else "good" if i % 3 == 1 else "average"}.',
            'product_category': ['electronics', 'clothing', 'books', 'home'][i % 4],
            'purchase_amount': round(50 + (i % 500), 2)
        })
    
    df = pd.DataFrame(large_data)
    df.to_csv('large_feedback_dataset.csv', index=False)
    print(f"Created large dataset with {len(df)} feedback entries")
    
    # Prompt template for feedback analysis
    prompt_template = """
    Analyze this customer feedback:
    
    Customer: {customer_name}
    Product Category: {product_category}
    Purchase Amount: ${purchase_amount}
    Feedback: "{feedback}"
    
    Provide analysis including sentiment, key themes, and recommendations.
    """
    
    # Configuration optimized for large datasets
    config = ProcessingConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        max_num_seqs=64,
        batch_size=32,
        cache_dir="./large_dataset_cache",
        max_retries=3,
        checkpoint_interval=50,  # Checkpoint every 50*32 = 1600 items
        max_tokens=800
    )
    
    # Pre-render prompts
    rendered_prompts = PromptRenderer.pre_render_all_prompts(prompt_template, df)
    
    processor = AsyncVLLMProcessor(config)
    
    try:
        await processor.initialize_engine()
        
        print("Processing large dataset...")
        print("Note: If this crashes, restart and it will resume from cache")
        
        results = await processor.process_batch(rendered_prompts, ReviewAnalysis)
        
        # Save results
        df['analysis'] = results
        df.to_json('large_dataset_results.json', orient='records', indent=2)
        
        # Final statistics
        stats = processor.stats.get_progress_stats()
        print(f"\nFinal Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
            
    except KeyboardInterrupt:
        print("\nGraceful shutdown initiated...")
        stats = processor.stats.get_progress_stats()
        print(f"Progress at shutdown: {stats['progress']}")
        print("Run again to resume from cache")
        
    finally:
        await processor.cleanup()


async def resume_processing_example():
    """Example: Demonstrate resuming from cache after crash"""
    
    print("This demonstrates resuming processing after a crash...")
    print("The cache will contain previous results, so processing resumes efficiently")
    
    # Use same configuration as large dataset
    config = ProcessingConfig(
        cache_dir="./large_dataset_cache"  # Same cache directory
    )
    
    # Check cache status
    from vllm_processor import CacheManager
    cache_manager = CacheManager(config.cache_dir)
    
    print(f"Cache directory: {config.cache_dir}")
    print(f"Cached items: {len(cache_manager.cache)}")
    
    # If there are cached items, show we can resume
    if len(cache_manager.cache) > 0:
        print("Found cached results - processing would resume from where it left off")
    else:
        print("No cached results found - would start fresh processing")
    
    cache_manager.close()


def main():
    """Main function to run examples"""
    print("vLLM Async Processor Examples")
    print("=" * 40)
    
    print("\n1. Processing product reviews (small dataset)")
    asyncio.run(process_product_reviews())
    
    print("\n" + "="*40)
    print("2. Processing large dataset with caching")
    # Uncomment to run large dataset example
    # asyncio.run(process_large_dataset())
    
    print("\n" + "="*40)
    print("3. Cache resume example")
    asyncio.run(resume_processing_example())


if __name__ == "__main__":
    main()

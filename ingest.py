import os
import json
import asyncio
import hashlib
from typing import List, Optional
from datetime import datetime
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# Setup logging
setup_logger("lightrag", level="INFO")

class LightRAGIngestion:
    def __init__(self, working_dir: str = "./lightrag_storage", use_vllm: bool = True):
        self.working_dir = working_dir
        self.use_vllm = use_vllm
        self.processed_ids_file = os.path.join(working_dir, "processed_texts.json")
        
        # Ensure working directory exists
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
        
        # Load previously processed text hashes
        self.processed_hashes = self._load_processed_hashes()
        
        # vLLM Configuration
        if use_vllm:
            self.VLLM_CHAT_BASE_URL = "http://localhost:8000/v1"
            self.VLLM_CHAT_API_KEY = "EMPTY"
            self.VLLM_CHAT_MODEL = "your-chat-model-name"  # Update this to your actual model
            
            self.VLLM_EMBEDDING_BASE_URL = "http://localhost:8001/v1"
            self.VLLM_EMBEDDING_API_KEY = "EMPTY"
            self.VLLM_EMBEDDING_MODEL = "your-embedding-model-name"  # Update this to your actual model
            self.EMBEDDING_DIMENSION = 768  # Update based on your embedding model's dimension
    
    def _load_processed_hashes(self) -> set:
        """Load previously processed text hashes"""
        if os.path.exists(self.processed_ids_file):
            try:
                with open(self.processed_ids_file, 'r') as f:
                    return set(json.load(f))
            except:
                return set()
        return set()
    
    def _save_processed_hashes(self):
        """Save processed text hashes"""
        with open(self.processed_ids_file, 'w') as f:
            json.dump(list(self.processed_hashes), f, indent=2)
    
    def _generate_text_hash(self, text: str) -> str:
        """Generate a unique hash for a text based on its content"""
        # Normalize text for consistent hashing (remove extra whitespace)
        normalized_text = ' '.join(text.split())
        return hashlib.sha256(normalized_text.encode()).hexdigest()[:16]
    
    async def _llm_model_func(self, prompt, system_prompt=None, history_messages=[], **kwargs):
        """LLM function using vLLM or OpenAI"""
        if self.use_vllm:
            return await openai_complete_if_cache(
                model=self.VLLM_CHAT_MODEL,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=self.VLLM_CHAT_API_KEY,
                base_url=self.VLLM_CHAT_BASE_URL,
                **kwargs
            )
        else:
            # Use OpenAI directly (make sure OPENAI_API_KEY is set)
            return await openai_complete_if_cache(
                model="gpt-4o-mini",
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs
            )
    
    async def _embedding_func(self, texts: list[str]):
        """Embedding function using vLLM or OpenAI"""
        if self.use_vllm:
            return await openai_embed(
                texts,
                model=self.VLLM_EMBEDDING_MODEL,
                api_key=self.VLLM_EMBEDDING_API_KEY,
                base_url=self.VLLM_EMBEDDING_BASE_URL
            )
        else:
            # Use OpenAI directly
            return await openai_embed(
                texts,
                model="text-embedding-3-small"
            )
    
    async def initialize_rag(self):
        """Initialize LightRAG instance"""
        embedding_dim = self.EMBEDDING_DIMENSION if self.use_vllm else 1536
        
        rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=self._llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=8192,
                func=self._embedding_func
            ),
            # Optional: Configure batch processing and entity extraction
            addon_params={
                "insert_batch_size": 10,  # Process 10 texts per batch
                "entity_types": ["person", "organization", "location", "event", "product", "technology"],
                "enable_llm_cache": True  # Cache LLM responses for efficiency
            }
        )
        
        await rag.initialize_storages()
        await initialize_pipeline_status()
        return rag
    
    async def ingest_text_array(self, json_file_path: str, min_text_length: int = 50):
        """
        Ingest an array of text strings from a JSON file with deduplication
        
        Args:
            json_file_path: Path to JSON file containing array of text strings
            min_text_length: Minimum length of text to process (filters out empty/short strings)
        """
        print(f"🚀 Starting text ingestion from {json_file_path}")
        
        # Load JSON data
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"❌ File not found: {json_file_path}")
            return
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON format: {e}")
            return
        
        # Ensure data is a list of strings
        if not isinstance(data, list):
            print("❌ JSON file must contain an array of text strings")
            return
        
        # Filter and validate texts
        valid_texts = []
        for text in data:
            if isinstance(text, str) and len(text.strip()) >= min_text_length:
                valid_texts.append(text.strip())
        
        print(f"📊 Found {len(valid_texts)} valid texts (filtered from {len(data)} total)")
        
        if not valid_texts:
            print("❌ No valid texts found to process")
            return
        
        # Check for new texts (deduplication)
        new_texts = []
        new_hashes = []
        duplicate_count = 0
        
        for text in valid_texts:
            text_hash = self._generate_text_hash(text)
            
            if text_hash not in self.processed_hashes:
                new_texts.append(text)
                new_hashes.append(text_hash)
            else:
                duplicate_count += 1
        
        if not new_texts:
            print(f"✅ All texts have already been processed ({duplicate_count} duplicates found)")
            return
        
        print(f"🆕 Found {len(new_texts)} new texts to process ({duplicate_count} duplicates skipped)")
        
        # Initialize RAG
        rag = await self.initialize_rag()
        
        try:
            # Process texts in batches
            batch_size = 10  # You can adjust this based on your needs
            total_processed = 0
            
            print(f"📝 Inserting texts into LightRAG (batch size: {batch_size})...")
            
            for i in range(0, len(new_texts), batch_size):
                batch_texts = new_texts[i:i + batch_size]
                batch_hashes = new_hashes[i:i + batch_size]
                
                # Insert batch with custom IDs
                await rag.ainsert(batch_texts, ids=batch_hashes)
                
                total_processed += len(batch_texts)
                print(f"   Processed {total_processed}/{len(new_texts)} texts...")
            
            # Update processed hashes
            self.processed_hashes.update(new_hashes)
            self._save_processed_hashes()
            
            print(f"\n✅ Successfully ingested {len(new_texts)} new texts!")
            
            # Print statistics
            print("\n📈 Ingestion Statistics:")
            print(f"   Total texts in file: {len(data)}")
            print(f"   Valid texts (>= {min_text_length} chars): {len(valid_texts)}")
            print(f"   New texts processed: {len(new_texts)}")
            print(f"   Duplicates skipped: {duplicate_count}")
            print(f"   Total texts in knowledge base: {len(self.processed_hashes)}")
            
        except Exception as e:
            print(f"❌ Error during ingestion: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await rag.finalize_storages()
    
    async def get_statistics(self):
        """Get statistics about the ingested texts"""
        print("\n📊 Knowledge Base Statistics:")
        print(f"   Total unique texts ingested: {len(self.processed_hashes)}")
        print(f"   Storage location: {self.working_dir}")
        
        # Check for graph file
        graph_file = os.path.join(self.working_dir, "graph_chunk_entity_relation.graphml")
        if os.path.exists(graph_file):
            file_size = os.path.getsize(graph_file) / (1024 * 1024)  # Convert to MB
            print(f"   Knowledge graph size: {file_size:.2f} MB")
    
    async def reset_ingestion(self):
        """Reset all ingestion data (use with caution!)"""
        print("⚠️  Resetting all ingestion data...")
        
        # Clear processed hashes
        self.processed_hashes.clear()
        self._save_processed_hashes()
        
        # Clear LightRAG storage
        import shutil
        if os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)
            os.makedirs(self.working_dir)
        
        print("✅ Ingestion data reset complete")


async def main():
    """Example usage"""
    # Configuration
    ingestion = LightRAGIngestion(
        working_dir="./my_lightrag_data",
        use_vllm=False  # Set to True if you have vLLM servers running
    )
    
    # Create example data file
    sample_texts = [
        "Alice Johnson is a senior software engineer at TechCorp Inc, specializing in machine learning and cloud architecture. She leads the AI Assistant Platform project and has 8 years of experience in the field.",
        
        "The AI Assistant Platform is an advanced enterprise solution that provides natural language query processing, multi-modal document analysis, and real-time collaboration features. It's built using LightRAG, FastAPI, and PostgreSQL.",
        
        "Bob Chen works as a product manager at TechCorp Inc. He collaborates closely with Alice Johnson on the AI Assistant Platform, where he serves as the product owner. Bob has expertise in product strategy and user research.",
        
        "",  # Empty string - will be filtered out
        
        "TechCorp Inc is a leading technology company founded in 2015, headquartered in San Francisco. The company has 250 employees and focuses on AI-powered enterprise solutions with an annual revenue of $50M.",
        
        "Short text",  # Too short - will be filtered out
        
        "The Knowledge Graph Integration project aims to integrate LightRAG with existing systems and improve query response time by 50%. The project team includes Alice Johnson and Carol Smith, with a budget of $500,000.",
        
        "Carol Smith is a data scientist in the Research department at TechCorp Inc, based in New York. She specializes in deep learning and NLP, leading the Knowledge Graph Research project using Neo4j and PyTorch."
    ]
    
    # Save sample data to file
    with open("sample_texts.json", "w", encoding='utf-8') as f:
        json.dump(sample_texts, f, indent=2, ensure_ascii=False)
    
    print("📝 Created sample_texts.json with example data")
    
    # Ingest texts
    await ingestion.ingest_text_array("sample_texts.json", min_text_length=50)
    
    # Get statistics
    await ingestion.get_statistics()
    
    # Test deduplication by running again
    print("\n🔄 Testing deduplication - running ingestion again...")
    await ingestion.ingest_text_array("sample_texts.json", min_text_length=50)
    
    # Example: Add more texts
    additional_texts = [
        "The company recently launched DataFlow Analytics, a new product for real-time data processing and visualization. It integrates seamlessly with the AI Assistant Platform.",
        
        "A new partnership has been announced between TechCorp Inc and CloudProvider X for enhanced cloud infrastructure support."
    ]
    
    with open("additional_texts.json", "w", encoding='utf-8') as f:
        json.dump(additional_texts, f, indent=2, ensure_ascii=False)
    
    print("\n📝 Adding additional texts...")
    await ingestion.ingest_text_array("additional_texts.json")


if __name__ == "__main__":
    # Check configuration
    if "your-chat-model-name" in ["your-chat-model-name", "your-embedding-model-name"]:
        print("⚠️  To use vLLM, update the model names in the script!")
        print("   Set use_vllm=False to use OpenAI instead")
        print("   Make sure OPENAI_API_KEY is set in your environment")
    
    asyncio.run(main())



import os
import asyncio
from typing import Optional, List, Dict
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# Setup logging
setup_logger("lightrag", level="INFO")

class LightRAGQuery:
    def __init__(self, working_dir: str = "./lightrag_storage", use_vllm: bool = True):
        self.working_dir = working_dir
        self.use_vllm = use_vllm
        self.rag = None
        
        # vLLM Configuration
        if use_vllm:
            self.VLLM_CHAT_BASE_URL = "http://localhost:8000/v1"
            self.VLLM_CHAT_API_KEY = "EMPTY"
            self.VLLM_CHAT_MODEL = "your-chat-model-name"  # Update this to your actual model
            
            self.VLLM_EMBEDDING_BASE_URL = "http://localhost:8001/v1"
            self.VLLM_EMBEDDING_API_KEY = "EMPTY"
            self.VLLM_EMBEDDING_MODEL = "your-embedding-model-name"  # Update this to your actual model
            self.EMBEDDING_DIMENSION = 768  # Update based on your embedding model's dimension
    
    async def _llm_model_func(self, prompt, system_prompt=None, history_messages=[], **kwargs):
        """LLM function using vLLM or OpenAI"""
        if self.use_vllm:
            return await openai_complete_if_cache(
                model=self.VLLM_CHAT_MODEL,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=self.VLLM_CHAT_API_KEY,
                base_url=self.VLLM_CHAT_BASE_URL,
                **kwargs
            )
        else:
            # Use OpenAI directly
            return await openai_complete_if_cache(
                model="gpt-4o-mini",
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs
            )
    
    async def _embedding_func(self, texts: list[str]):
        """Embedding function using vLLM or OpenAI"""
        if self.use_vllm:
            return await openai_embed(
                texts,
                model=self.VLLM_EMBEDDING_MODEL,
                api_key=self.VLLM_EMBEDDING_API_KEY,
                base_url=self.VLLM_EMBEDDING_BASE_URL
            )
        else:
            # Use OpenAI directly
            return await openai_embed(
                texts,
                model="text-embedding-3-small"
            )
    
    async def initialize(self):
        """Initialize LightRAG instance"""
        if self.rag is not None:
            return  # Already initialized
        
        embedding_dim = self.EMBEDDING_DIMENSION if self.use_vllm else 1536
        
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=self._llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=8192,
                func=self._embedding_func
            )
        )
        
        await self.rag.initialize_storages()
        await initialize_pipeline_status()
    
    async def query(
        self,
        question: str,
        mode: str = "hybrid",
        top_k: int = 20,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        custom_prompt: Optional[str] = None
    ):
        """
        Query the LightRAG knowledge base
        
        Args:
            question: The query string
            mode: Query mode - "local", "global", "hybrid", "mix", "naive"
            top_k: Number of top results to retrieve
            conversation_history: Optional conversation history for context
            stream: Whether to stream the response
            custom_prompt: Optional custom prompt for formatting the response
        
        Returns:
            Query response
        """
        # Ensure RAG is initialized
        await self.initialize()
        
        # Build query parameters
        query_params = QueryParam(
            mode=mode,
            top_k=top_k,
            stream=stream
        )
        
        # Add conversation history if provided
        if conversation_history:
            query_params.conversation_history = conversation_history
        
        # Add custom prompt if provided
        if custom_prompt:
            query_params.user_prompt = custom_prompt
        
        # Execute query
        result = await self.rag.aquery(question, param=query_params)
        
        return result
    
    async def multi_mode_query(self, question: str, modes: List[str] = None):
        """
        Query using multiple modes and compare results
        
        Args:
            question: The query string
            modes: List of modes to query (default: all modes)
        
        Returns:
            Dictionary with results from each mode
        """
        if modes is None:
            modes = ["local", "global", "hybrid", "mix", "naive"]
        
        results = {}
        
        for mode in modes:
            print(f"\n🔍 Querying in {mode.upper()} mode...")
            try:
                result = await self.query(question, mode=mode)
                results[mode] = result
                print(f"✅ {mode.upper()} mode completed")
            except Exception as e:
                results[mode] = f"Error: {str(e)}"
                print(f"❌ {mode.upper()} mode failed: {e}")
        
        return results
    
    async def interactive_query(self):
        """Run an interactive query session"""
        await self.initialize()
        
        print("🤖 LightRAG Interactive Query Session")
        print("Commands:")
        print("  - Type your question and press Enter")
        print("  - Type 'mode:<mode>' to change query mode (local/global/hybrid/mix/naive)")
        print("  - Type 'compare' to compare all modes for the last question")
        print("  - Type 'quit' to exit")
        print("-" * 50)
        
        current_mode = "hybrid"
        last_question = None
        conversation_history = []
        
        while True:
            try:
                user_input = input(f"\n[Mode: {current_mode}] Your question: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower().startswith('mode:'):
                    new_mode = user_input[5:].strip().lower()
                    if new_mode in ["local", "global", "hybrid", "mix", "naive"]:
                        current_mode = new_mode
                        print(f"✅ Switched to {current_mode.upper()} mode")
                    else:
                        print("❌ Invalid mode. Choose from: local, global, hybrid, mix, naive")
                    continue
                
                if user_input.lower() == 'compare' and last_question:
                    print(f"\n📊 Comparing all modes for: '{last_question}'")
                    results = await self.multi_mode_query(last_question)
                    
                    for mode, result in results.items():
                        print(f"\n{'='*20} {mode.upper()} MODE {'='*20}")
                        print(result)
                        print("="*50)
                    continue
                
                # Regular query
                last_question = user_input
                print(f"\n💭 Processing query in {current_mode.upper()} mode...")
                
                result = await self.query(
                    user_input,
                    mode=current_mode,
                    conversation_history=conversation_history
                )
                
                print(f"\n📝 Response:\n{result}")
                
                # Update conversation history
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": result})
                
                # Keep only last 3 turns
                if len(conversation_history) > 6:
                    conversation_history = conversation_history[-6:]
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    async def finalize(self):
        """Clean up resources"""
        if self.rag:
            await self.rag.finalize_storages()


async def main():
    """Example usage of the query system with text-based knowledge base"""
    # Initialize query system
    query_system = LightRAGQuery(
        working_dir="./my_lightrag_data",
        use_vllm=False  # Set to True if you have vLLM servers running
    )
    
    try:
        # Example 1: Query about people
        print("📋 Example 1: Query about People")
        result = await query_system.query(
            "Who are the key people mentioned and what are their roles?",
            mode="hybrid"
        )
        print(f"Result: {result}\n")
        
        # Example 2: Query about relationships
        print("📋 Example 2: Query about Relationships")
        result = await query_system.query(
            "How are Alice Johnson and Bob Chen connected?",
            mode="local"
        )
        print(f"Result: {result}\n")
        
        # Example 3: Query about projects
        print("📋 Example 3: Query about Projects")
        result = await query_system.query(
            "What projects are mentioned and what are their objectives?",
            mode="hybrid",
            custom_prompt="Provide a structured summary with project names, teams, and key objectives"
        )
        print(f"Result: {result}\n")
        
        # Example 4: Compare different modes
        print("📋 Example 4: Multi-mode Comparison")
        results = await query_system.multi_mode_query(
            "What is TechCorp Inc and what does it do?"
        )
        
        for mode, result in results.items():
            print(f"\n--- {mode.upper()} Mode ---")
            print(result[:300] + "..." if len(result) > 300 else result)
        
        # Example 5: Interactive session
        print("\n📋 Starting Interactive Session...")
        print("Try questions like:")
        print("- What technologies are mentioned?")
        print("- Who works on what projects?")
        print("- What is the relationship between the company and its products?")
        await query_system.interactive_query()
        
    finally:
        await query_system.finalize()


if __name__ == "__main__":
    # For vLLM usage, update the model names first
    if "your-chat-model-name" in ["your-chat-model-name", "your-embedding-model-name"]:
        print("⚠️  To use vLLM, update the model names in the script!")
        print("   Set use_vllm=False to use OpenAI instead")
    
    asyncio.run(main())



"""
Complete example demonstrating the full workflow of ingesting and querying
text-based narratives with LightRAG.
"""

import asyncio
import json
import os
from lightrag_ingestion import LightRAGIngestion
from lightrag_query import LightRAGQuery


async def create_sample_data():
    """Create sample text data files for demonstration"""
    
    # Sample narrative texts about a tech company
    company_texts = [
        "TechVision AI is a pioneering artificial intelligence company founded in 2019 by Dr. Emily Chen and Mark Thompson. The company specializes in developing advanced natural language processing systems and knowledge graph technologies. Their headquarters is located in Seattle, with additional offices in London and Tokyo.",
        
        "Dr. Emily Chen serves as the CEO and Chief Scientist at TechVision AI. She holds a PhD in Computer Science from Stanford University and has published over 50 papers on machine learning and natural language understanding. Before founding TechVision, she worked as a principal researcher at Google DeepMind.",
        
        "Mark Thompson is the CTO of TechVision AI, bringing 15 years of experience in scalable systems architecture. He previously led engineering teams at Amazon Web Services and Microsoft Azure. Mark is responsible for the company's technical infrastructure and leads a team of 40 engineers.",
        
        "The flagship product of TechVision AI is called KnowledgeGraph Pro, an enterprise solution that automatically constructs and maintains knowledge graphs from unstructured documents. The system uses advanced NLP models to extract entities, relationships, and semantic connections from text data.",
        
        "In 2023, TechVision AI secured $50 million in Series B funding led by Venture Capital Partners, with participation from Tech Innovations Fund and AI Future Investments. The funding will be used to expand the engineering team and accelerate product development.",
        
        "Sarah Rodriguez joined TechVision AI as VP of Product in 2022. She works closely with Dr. Chen and the research team to translate cutting-edge AI research into practical product features. Sarah previously worked at Salesforce where she led the AI product strategy team.",
        
        "The KnowledgeGraph Pro system has been adopted by several Fortune 500 companies including GlobalBank, HealthTech Corp, and Retail Giants Inc. These clients use the system to analyze vast amounts of internal documentation and extract actionable insights.",
        
        "TechVision AI's research team, led by Dr. Chen, recently published a breakthrough paper on 'Hierarchical Entity Recognition in Long Documents' at the NeurIPS 2023 conference. The paper introduces novel techniques that improve entity extraction accuracy by 35%.",
        
        "The company culture at TechVision AI emphasizes innovation, collaboration, and continuous learning. They offer employees dedicated research time, similar to Google's 20% time policy, encouraging exploration of new ideas and technologies.",
        
        "In partnership with the University of Washington, TechVision AI established an AI research lab focused on explainable AI and ethical knowledge graph construction. The lab is co-directed by Dr. Chen and Professor James Liu from UW's Computer Science department."
    ]
    
    # Sample texts about projects and technical details
    project_texts = [
        "Project Apollo is TechVision AI's initiative to create a multi-modal knowledge graph system that can process not just text, but also images, tables, and diagrams. The project team includes 15 engineers and 5 research scientists, with an expected launch date in Q4 2024.",
        
        "The core technology behind KnowledgeGraph Pro uses a combination of transformer-based language models and graph neural networks. The system can process documents in over 20 languages and maintains consistency across multilingual knowledge bases.",
        
        "TechVision AI's infrastructure runs on a hybrid cloud architecture using AWS and Google Cloud Platform. The system processes over 10 million documents daily for their enterprise clients, with an average response time of under 2 seconds for complex queries.",
        
        "Dr. Michael Park leads the Machine Learning Infrastructure team at TechVision AI. His team has developed a proprietary model training pipeline that reduces training time by 60% compared to standard approaches, enabling rapid iteration on new models.",
        
        "The company's latest research focuses on temporal knowledge graphs that can track how entities and relationships change over time. This feature is particularly valuable for clients in the financial and healthcare sectors who need to understand evolving patterns."
    ]
    
    # Save to JSON files
    with open("company_narratives.json", "w", encoding="utf-8") as f:
        json.dump(company_texts, f, indent=2, ensure_ascii=False)
    
    with open("project_narratives.json", "w", encoding="utf-8") as f:
        json.dump(project_texts, f, indent=2, ensure_ascii=False)
    
    print("✅ Created sample data files: company_narratives.json and project_narratives.json")


async def demonstrate_workflow():
    """Demonstrate the complete workflow"""
    
    # Configuration
    working_dir = "./techvision_knowledge_base"
    use_vllm = False  # Set to True if you have vLLM servers running
    
    print("🚀 LightRAG Text-based Workflow Demonstration")
    print("=" * 60)
    
    # Step 1: Create sample data
    print("\n📝 Step 1: Creating sample narrative data...")
    await create_sample_data()
    
    # Step 2: Initialize ingestion system
    print("\n🔧 Step 2: Initializing ingestion system...")
    ingestion = LightRAGIngestion(
        working_dir=working_dir,
        use_vllm=use_vllm
    )
    
    # Step 3: Ingest the narratives
    print("\n📥 Step 3: Ingesting narrative texts...")
    
    # Ingest company narratives
    print("\n--- Ingesting company narratives ---")
    await ingestion.ingest_text_array("company_narratives.json")
    
    # Ingest project narratives
    print("\n--- Ingesting project narratives ---")
    await ingestion.ingest_text_array("project_narratives.json")
    
    # Show statistics
    await ingestion.get_statistics()
    
    # Step 4: Query the knowledge base
    print("\n🔍 Step 4: Querying the knowledge base...")
    
    query_system = LightRAGQuery(
        working_dir=working_dir,
        use_vllm=use_vllm
    )
    
    # Example queries
    example_queries = [
        {
            "question": "Who are the founders of TechVision AI and what are their backgrounds?",
            "mode": "hybrid",
            "description": "Finding specific information about people"
        },
        {
            "question": "What is KnowledgeGraph Pro and which companies use it?",
            "mode": "local",
            "description": "Product information and client relationships"
        },
        {
            "question": "How is TechVision AI's technical infrastructure organized?",
            "mode": "hybrid",
            "description": "Technical architecture and team structure"
        },
        {
            "question": "What are the main research initiatives and partnerships?",
            "mode": "global",
            "description": "High-level research themes"
        },
        {
            "question": "Explain the relationship between Dr. Emily Chen, Mark Thompson, and the various projects at TechVision AI",
            "mode": "hybrid",
            "description": "Complex relationship query"
        }
    ]
    
    for i, query_info in enumerate(example_queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query_info['description']}")
        print(f"Question: {query_info['question']}")
        print(f"Mode: {query_info['mode'].upper()}")
        print("-" * 60)
        
        result = await query_system.query(
            query_info['question'],
            mode=query_info['mode']
        )
        
        print(f"Answer:\n{result}")
    
    # Step 5: Demonstrate deduplication
    print("\n\n🔄 Step 5: Demonstrating deduplication...")
    print("Re-ingesting the same files to show deduplication in action:")
    
    await ingestion.ingest_text_array("company_narratives.json")
    
    # Step 6: Add new information
    print("\n\n➕ Step 6: Adding new information...")
    
    new_texts = [
        "In late 2023, TechVision AI announced a strategic partnership with CloudScale Systems to enhance their distributed processing capabilities. This partnership will allow KnowledgeGraph Pro to handle petabyte-scale datasets.",
        
        "Dr. Lisa Wang joined TechVision AI as Head of Ethics and Responsible AI in January 2024. She is working with Dr. Chen to ensure that all AI systems developed by the company adhere to strict ethical guidelines and transparency standards.",
        
        "TechVision AI's Seattle office is expanding with a new 50,000 square foot facility that will house the growing research team. The new space includes dedicated labs for experimental AI research and collaboration areas for cross-functional teams."
    ]
    
    with open("updates.json", "w", encoding="utf-8") as f:
        json.dump(new_texts, f, indent=2, ensure_ascii=False)
    
    print("Created updates.json with new information")
    await ingestion.ingest_text_array("updates.json")
    
    # Query the new information
    print("\n🔍 Querying about the new information...")
    result = await query_system.query(
        "What are the recent developments and new hires at TechVision AI?",
        mode="hybrid"
    )
    print(f"Answer:\n{result}")
    
    # Step 7: Multi-mode comparison
    print("\n\n🔬 Step 7: Comparing different query modes...")
    comparison_question = "What is the organizational structure and key personnel at TechVision AI?"
    
    print(f"Question: {comparison_question}")
    results = await query_system.multi_mode_query(comparison_question, modes=["local", "global", "hybrid"])
    
    for mode, result in results.items():
        print(f"\n--- {mode.upper()} Mode ---")
        print(result[:400] + "..." if len(result) > 400 else result)
    
    # Finalize
    await query_system.finalize()
    
    print("\n\n✅ Demonstration complete!")
    print("\n💡 Next steps:")
    print("1. Try the interactive query mode: await query_system.interactive_query()")
    print("2. Add your own narrative texts to explore different domains")
    print("3. Experiment with different query modes for various question types")
    print("4. Use custom prompts to format responses for specific needs")


async def quick_start():
    """Quick start example with minimal setup"""
    print("⚡ Quick Start Example")
    
    # Create simple test data
    test_texts = [
        "Alice is a software engineer who works on the AI Assistant project. She specializes in natural language processing and has been with the company for 5 years.",
        "The AI Assistant project uses LightRAG for knowledge management. It can answer complex questions by understanding relationships between different pieces of information.",
        "Bob is the project manager for AI Assistant. He coordinates between Alice and the other developers to ensure timely delivery of features."
    ]
    
    with open("quick_test.json", "w") as f:
        json.dump(test_texts, f)
    
    # Ingest
    ingestion = LightRAGIngestion("./quick_test_kb", use_vllm=False)
    await ingestion.ingest_text_array("quick_test.json")
    
    # Query
    query_system = LightRAGQuery("./quick_test_kb", use_vllm=False)
    result = await query_system.query("Who works on the AI Assistant project and what are their roles?")
    print(f"\nAnswer: {result}")
    
    await query_system.finalize()


async def main():
    """Main entry point"""
    print("🎯 LightRAG Text-based Knowledge Base System")
    print("Choose an option:")
    print("1. Run complete demonstration")
    print("2. Quick start example")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        await demonstrate_workflow()
    elif choice == "2":
        await quick_start()
    else:
        print("Exiting...")


if __name__ == "__main__":
    # Ensure OpenAI API key is set if not using vLLM
    if not os.getenv("OPENAI_API_KEY") and not os.path.exists(".env"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
        print("   Or use vLLM by setting use_vllm=True\n")
    
    asyncio.run(main())

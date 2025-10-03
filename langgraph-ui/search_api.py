"""
Main search API interface for the advanced search engine.
"""
import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path

# Optional: FastAPI for REST API
try:
    from fastapi import FastAPI, HTTPException, Query
    from pydantic import BaseModel, Field
    from typing import Literal
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. REST API features will be disabled.")

from config import SearchConfig
from engine import get_search_engine
from utils import logger, PerformanceMonitor


class SearchRequest(BaseModel):
    """Search request model."""
    query: Union[str, List[str]] = Field(..., description="Query string or list of queries")
    mode: Literal["keyword", "semantic", "hybrid"] = Field("hybrid", description="Search mode")
    top_k: int = Field(SearchConfig.DEFAULT_TOP_K, description="Number of results to return")
    fields: Optional[List[str]] = Field(None, description="Specific fields for keyword search")
    keyword_weight: float = Field(1.0, description="Weight for keyword search in hybrid mode")
    semantic_weight: float = Field(1.0, description="Weight for semantic search in hybrid mode")


class SearchResponse(BaseModel):
    """Search response model."""
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    count: int = Field(..., description="Number of results")
    search_time: float = Field(..., description="Search time in seconds")
    mode: str = Field(..., description="Search mode used")


class SearchAPI:
    """Main search API class."""
    
    def __init__(self):
        self.engine = get_search_engine()
        self.monitor = PerformanceMonitor()
    
    async def search(self,
                    query: Union[str, List[str]],
                    mode: str = "hybrid",
                    top_k: int = SearchConfig.DEFAULT_TOP_K,
                    fields: Optional[List[str]] = None,
                    keyword_weight: float = 1.0,
                    semantic_weight: float = 1.0) -> SearchResponse:
        """
        Perform search with specified parameters.
        
        Args:
            query: Query string or list of queries
            mode: Search mode - "keyword", "semantic", or "hybrid"
            top_k: Number of results to return
            fields: Specific fields for keyword search
            keyword_weight: Weight for keyword search in hybrid mode
            semantic_weight: Weight for semantic search in hybrid mode
        
        Returns:
            SearchResponse with results and metadata
        """
        start_time = time.time()
        
        try:
            # Perform search based on mode
            if mode == "keyword":
                results = self.engine.keyword_search(
                    query, 
                    top_k=top_k, 
                    fields=fields
                )
            elif mode == "semantic":
                if isinstance(query, list):
                    query = query[0]
                results = await self.engine.semantic_search(
                    query, 
                    top_k=top_k
                )
            elif mode == "hybrid":
                results = await self.engine.hybrid_search(
                    query,
                    top_k=top_k,
                    keyword_weight=keyword_weight,
                    semantic_weight=semantic_weight
                )
            else:
                raise ValueError(f"Invalid search mode: {mode}")
            
            search_time = time.time() - start_time
            
            # Log performance
            if search_time > SearchConfig.SEARCH_TIMEOUT:
                logger.warning(f"Search exceeded timeout: {search_time:.3f}s > {SearchConfig.SEARCH_TIMEOUT}s")
            else:
                logger.info(f"Search completed in {search_time:.3f}s")
            
            return SearchResponse(
                results=results,
                count=len(results),
                search_time=search_time,
                mode=mode
            )
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def batch_search(self,
                          queries: List[str],
                          mode: str = "hybrid",
                          top_k: int = SearchConfig.DEFAULT_TOP_K) -> List[SearchResponse]:
        """
        Perform batch search for multiple queries.
        
        Args:
            queries: List of query strings
            mode: Search mode
            top_k: Number of results per query
        
        Returns:
            List of SearchResponse objects
        """
        results = []
        
        # Process queries concurrently
        tasks = []
        for query in queries:
            task = asyncio.create_task(
                self.search(query, mode=mode, top_k=top_k)
            )
            tasks.append(task)
        
        # Wait for all searches to complete
        responses = await asyncio.gather(*tasks)
        
        return responses


# Create FastAPI app if available
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Advanced Search Engine API",
        description="High-performance search engine with keyword, semantic, and hybrid search",
        version="1.0.0"
    )
    
    search_api = SearchAPI()
    
    @app.post("/search", response_model=SearchResponse)
    async def search_endpoint(request: SearchRequest):
        """Execute a search query."""
        try:
            response = await search_api.search(
                query=request.query,
                mode=request.mode,
                top_k=request.top_k,
                fields=request.fields,
                keyword_weight=request.keyword_weight,
                semantic_weight=request.semantic_weight
            )
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/batch_search", response_model=List[SearchResponse])
    async def batch_search_endpoint(
        queries: List[str] = Query(..., description="List of queries"),
        mode: str = Query("hybrid", description="Search mode"),
        top_k: int = Query(SearchConfig.DEFAULT_TOP_K, description="Results per query")
    ):
        """Execute multiple search queries in batch."""
        try:
            responses = await search_api.batch_search(
                queries=queries,
                mode=mode,
                top_k=top_k
            )
            return responses
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health")
    async def health_check():
        """Check if the search engine is healthy."""
        return {"status": "healthy", "indices": {
            "tantivy": search_api.engine.tantivy_searcher is not None,
            "faiss": search_api.engine.faiss_searcher is not None
        }}
    
    @app.get("/stats")
    async def get_stats():
        """Get search engine statistics."""
        stats = {
            "config": SearchConfig.to_dict(),
            "indices": {
                "tantivy": {
                    "available": search_api.engine.tantivy_searcher is not None,
                    "path": str(SearchConfig.TANTIVY_INDEX_PATH)
                },
                "faiss": {
                    "available": search_api.engine.faiss_searcher is not None,
                    "path": str(SearchConfig.FAISS_INDEX_PATH),
                    "vectors": search_api.engine.faiss_searcher.index.ntotal 
                              if search_api.engine.faiss_searcher else 0
                }
            }
        }
        return stats


# CLI interface
async def cli_search():
    """Command-line interface for search."""
    search_api = SearchAPI()
    
    print("Advanced Search Engine CLI")
    print("-" * 50)
    print("Available modes: keyword, semantic, hybrid")
    print("Type 'exit' to quit")
    print("-" * 50)
    
    while True:
        try:
            # Get query
            query = input("\nEnter search query: ").strip()
            if query.lower() == 'exit':
                break
            
            if not query:
                continue
            
            # Get mode
            mode = input("Search mode (keyword/semantic/hybrid) [hybrid]: ").strip() or "hybrid"
            
            # Get top_k
            top_k_str = input(f"Number of results [{SearchConfig.DEFAULT_TOP_K}]: ").strip()
            top_k = int(top_k_str) if top_k_str else SearchConfig.DEFAULT_TOP_K
            
            # Perform search
            print(f"\nSearching in {mode} mode...")
            response = await search_api.search(query, mode=mode, top_k=top_k)
            
            # Display results
            print(f"\nFound {response.count} results in {response.search_time:.3f}s:")
            print("-" * 50)
            
            for i, result in enumerate(response.results, 1):
                print(f"\n{i}. Score: {result['_score']:.4f} | Type: {result['_search_type']}")
                
                # Show first few fields
                for key, value in list(result.items())[:5]:
                    if not key.startswith('_'):
                        print(f"   {key}: {str(value)[:100]}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "server" and FASTAPI_AVAILABLE:
            # Run FastAPI server
            import uvicorn
            print(f"Starting search API server on http://localhost:8000")
            print("API docs available at http://localhost:8000/docs")
            uvicorn.run(app, host="0.0.0.0", port=8000)
        
        elif command == "cli":
            # Run CLI interface
            asyncio.run(cli_search())
        
        elif command == "test":
            # Run test search
            async def test():
                api = SearchAPI()
                
                # Test keyword search
                print("Testing keyword search...")
                response = await api.search(["test", "example"], mode="keyword", top_k=5)
                print(f"  Found {response.count} results in {response.search_time:.3f}s")
                
                # Test semantic search
                print("Testing semantic search...")
                response = await api.search("test query", mode="semantic", top_k=5)
                print(f"  Found {response.count} results in {response.search_time:.3f}s")
                
                # Test hybrid search
                print("Testing hybrid search...")
                response = await api.search("test query", mode="hybrid", top_k=5)
                print(f"  Found {response.count} results in {response.search_time:.3f}s")
                
                print("\nAll tests completed!")
            
            asyncio.run(test())
        
        else:
            print(f"Unknown command: {command}")
            print("Available commands: server, cli, test")
    
    else:
        print("Advanced Search Engine")
        print("-" * 50)
        print("Usage:")
        print("  python search.py server  - Start REST API server (requires FastAPI)")
        print("  python search.py cli     - Start interactive CLI")
        print("  python search.py test    - Run test searches")
        print("-" * 50)
        print("\nMake sure to run ingestion.py first to create indices!")


if __name__ == "__main__":
    main()

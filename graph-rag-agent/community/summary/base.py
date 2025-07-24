from abc import ABC, abstractmethod
from typing import List, Dict
from langchain_community.graphs import Neo4jGraph
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from model.get_models import get_llm_model
import concurrent.futures
import time

from config.settings import MAX_WORKERS

class BaseCommunityDescriber:
    """Community information formatting tool"""
    
    @staticmethod
    def prepare_string(data: Dict) -> str:
        """Convert community information to readable string"""
        try:
            nodes_str = "Nodes are:\n"
            for node in data.get('nodes', []):
                node_id = node.get('id', 'unknown_id')
                node_type = node.get('type', 'unknown_type')
                node_description = (
                    f", description: {node['description']}"
                    if 'description' in node and node['description']
                    else ""
                )
                nodes_str += f"id: {node_id}, type: {node_type}{node_description}\n"

            rels_str = "Relationships are:\n"
            for rel in data.get('rels', []):
                start = rel.get('start', 'unknown_start')
                end = rel.get('end', 'unknown_end')
                rel_type = rel.get('type', 'unknown_type')
                description = (
                    f", description: {rel['description']}"
                    if 'description' in rel and rel['description']
                    else ""
                )
                rels_str += f"({start})-[:{rel_type}]->({end}){description}\n"

            return nodes_str + "\n" + rels_str
        except Exception as e:
            print(f"Error formatting community information: {e}")
            return f"Error: {str(e)}\nData: {str(data)}"

class BaseCommunityRanker:
    """Community weight calculation tool"""
    
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
    
    def calculate_ranks(self) -> None:
        """Calculate community weights"""
        start_time = time.time()
        print("Calculating community weights...")
        
        try:
            result = self.graph.query("""
            MATCH (c:`__Community__`)<-[:IN_COMMUNITY*]-(:`__Entity__`)<-[:MENTIONS]-(d:`__Chunk__`)
            WITH c, count(distinct d) AS rank
            SET c.community_rank = rank
            RETURN count(c) AS processed_count
            """)
            
            processed_count = result[0]['processed_count'] if result else 0
            print(f"Community weight calculation completed, processed {processed_count} communities, "
                  f"time elapsed: {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error calculating community weights: {e}")
            self._calculate_ranks_fallback()
    
    def _calculate_ranks_fallback(self):
        """Fallback weight calculation method"""
        try:
            self.graph.query("""
            MATCH (c:`__Community__`)<-[:IN_COMMUNITY]-(e:`__Entity__`)
            WITH c, count(e) AS entity_count
            SET c.community_rank = entity_count
            """)
            print("Using entity count as community weight")
        except Exception as e:
            print(f"Fallback weight calculation also failed: {e}")

class BaseCommunityStorer:
    """Community information storage tool"""
    
    def __init__(self, graph: Neo4jGraph):
        self.graph = graph
    
    def store_summaries(self, summaries: List[Dict]) -> None:
        """Store community summaries"""
        if not summaries:
            print("No community summaries to store")
            return
            
        start_time = time.time()
        print(f"Starting to store {len(summaries)} community summaries...")
        
        batch_size = min(100, max(10, len(summaries) // 5))
        total_batches = (len(summaries) + batch_size - 1) // batch_size
        
        for i in range(0, len(summaries), batch_size):
            batch = summaries[i:i+batch_size]
            batch_start = time.time()
            
            try:
                self.graph.query("""
                UNWIND $data AS row
                MERGE (c:__Community__ {id:row.community})
                SET c.summary = row.summary, 
                    c.full_content = row.full_content,
                    c.summary_created_at = datetime()
                """, params={"data": batch})
                
                print(f"Stored batch {i//batch_size + 1}/{total_batches}, "
                      f"time elapsed: {time.time() - batch_start:.2f} seconds")
                
            except Exception as e:
                print(f"Error storing community summary batch: {e}")
                self._store_summaries_one_by_one(batch)
    
    def _store_summaries_one_by_one(self, summaries: List[Dict]):
        """Store community summaries one by one"""
        for summary in summaries:
            try:
                self.graph.query("""
                MERGE (c:__Community__ {id:$community})
                SET c.summary = $summary, 
                    c.full_content = $full_content,
                    c.summary_created_at = datetime()
                """, params=summary)
            except Exception as e:
                print(f"Error storing individual community summary: {e}")

class BaseSummarizer(ABC):
    """Base class for community summary generators"""
    
    def __init__(self, graph: Neo4jGraph):
        """Initialize base community summary generator"""
        self.graph = graph
        self.llm = get_llm_model()
        self.describer = BaseCommunityDescriber()
        self.ranker = BaseCommunityRanker(graph)
        self.storer = BaseCommunityStorer(graph)
        self._setup_llm_chain()
        
        # Performance monitoring
        self.llm_time = 0
        self.query_time = 0
        self.store_time = 0
        
        self.max_workers = MAX_WORKERS
        print(f"Community summary generator initialized, parallel threads: {self.max_workers}")

    def _setup_llm_chain(self) -> None:
        """Setup LLM processing chain"""
        try:
            community_prompt = ChatPromptTemplate.from_messages([
                ("system", "Given an input triple, generate an information summary. No preamble."),
                ("human", "{community_info}"),
            ])
            self.community_chain = community_prompt | self.llm | StrOutputParser()
        except Exception as e:
            print(f"Error setting up LLM processing chain: {e}")
            raise

    @abstractmethod
    def collect_community_info(self) -> List[Dict]:
        """Abstract method for collecting community information"""
        pass

    def process_communities(self) -> List[Dict]:
        """Process all communities"""
        total_start_time = time.time()
        print("Starting to process community summaries...")
        
        try:
            # Calculate community weights
            rank_start = time.time()
            self.ranker.calculate_ranks()
            rank_time = time.time() - rank_start
            
            # Collect community information
            query_start = time.time()
            community_info = self.collect_community_info()
            self.query_time = time.time() - query_start
            
            if not community_info:
                print("No communities found to process")
                return []
            
            # Generate summaries in parallel
            llm_start = time.time()
            optimal_workers = min(self.max_workers, max(1, len(community_info) // 2))
            print(f"Starting parallel generation of {len(community_info)} community summaries, "
                  f"using {optimal_workers} threads...")
            
            summaries = self._process_communities_parallel(
                community_info, 
                optimal_workers
            )
            
            self.llm_time = time.time() - llm_start
            
            # Save summaries
            store_start = time.time()
            self.storer.store_summaries(summaries)
            self.store_time = time.time() - store_start
            
            # Output performance statistics
            total_time = time.time() - total_start_time
            self._print_performance_stats(
                total_time, rank_time, 
                self.query_time, self.llm_time, 
                self.store_time
            )
            
            return summaries
            
        except Exception as e:
            print(f"Error processing community summaries: {str(e)}")
            raise
    
    def _process_communities_parallel(
        self, 
        community_info: List[Dict], 
        workers: int
    ) -> List[Dict]:
        """Process community summaries in parallel"""
        summaries = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_community = {
                executor.submit(self._process_single_community, info): i 
                for i, info in enumerate(community_info)
            }
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_community)):
                try:
                    result = future.result()
                    summaries.append(result)
                    
                    if (i+1) % 10 == 0 or (i+1) == len(community_info):
                        print(f"Processed {i+1}/{len(community_info)} "
                              f"({(i+1)/len(community_info)*100:.1f}%)")
                        
                except Exception as e:
                    print(f"Error processing community summary: {e}")
        
        return summaries
    
    def _process_single_community(self, community: Dict) -> Dict:
        """Process individual community summary"""
        community_id = community.get('communityId', 'unknown')
        
        try:
            stringify_info = self.describer.prepare_string(community)
            
            if len(stringify_info) < 10:
                print(f"Community {community_id} has too little information, skipping summary generation")
                return {
                    "community": community_id,
                    "summary": "This community does not have enough information to generate a summary.",
                    "full_content": stringify_info
                }
            
            summary = self.community_chain.invoke({'community_info': stringify_info})
            
            return {
                "community": community_id,
                "summary": summary,
                "full_content": stringify_info
            }
        except Exception as e:
            print(f"Error processing community {community_id} summary: {e}")
            return {
                "community": community_id,
                "summary": f"Error generating summary: {str(e)}",
                "full_content": str(community)
            }
    
    def _print_performance_stats(
        self, 
        total_time: float,
        rank_time: float,
        query_time: float,
        llm_time: float,
        store_time: float
    ) -> None:
        """Print performance statistics"""
        print(f"\nCommunity summary processing completed, total time: {total_time:.2f} seconds")
        print(f"  Community weight calculation: {rank_time:.2f} seconds ({rank_time/total_time*100:.1f}%)")
        print(f"  Community information query: {query_time:.2f} seconds ({query_time/total_time*100:.1f}%)")
        print(f"  Summary generation (LLM): {llm_time:.2f} seconds ({llm_time/total_time*100:.1f}%)")
        print(f"  Result storage: {store_time:.2f} seconds ({store_time/total_time*100:.1f}%)")
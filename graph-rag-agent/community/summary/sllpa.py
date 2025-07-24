from typing import List, Dict
from .base import BaseSummarizer
import time

from config.settings import BATCH_SIZE

class SLLPASummarizer(BaseSummarizer):
    """Community summary generator for SLLPA algorithm"""
    
    def collect_community_info(self) -> List[Dict]:
        """Collect SLLPA community information"""
        start_time = time.time()
        print("Collecting SLLPA community information...")
        
        try:
            # Get total community count
            count_result = self.graph.query("""
            MATCH (c:`__Community__`)
            WHERE c.level = 0  // All SLLPA communities are at level 0
            RETURN count(c) AS community_count
            """)
            
            community_count = count_result[0]['community_count'] if count_result else 0
            if not community_count:
                print("No SLLPA communities found")
                return []
                
            print(f"Found {community_count} SLLPA communities, starting to collect detailed information")
            
            if community_count > 1000:
                return self._collect_info_in_batches(community_count)
            
            result = self.graph.query("""
            MATCH (c:`__Community__`)
            WHERE c.level = 0
            WITH c ORDER BY c.community_rank DESC NULLS LAST
            LIMIT 200
            
            MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
            WITH c, collect(e) AS nodes
            WHERE size(nodes) > 1
            
            CALL {
                WITH nodes
                MATCH (n1:__Entity__)
                WHERE n1 IN nodes
                MATCH (n2:__Entity__)
                WHERE n2 IN nodes AND id(n1) < id(n2)
                MATCH (n1)-[r]->(n2)
                WITH collect(distinct r) as relationships
                LIMIT 100
                RETURN relationships
            }
            
            RETURN c.id AS communityId,
                   [n in nodes | {
                       id: n.id, 
                       description: n.description, 
                       type: [el in labels(n) WHERE el <> '__Entity__'][0]
                   }] AS nodes,
                   [r in relationships | {
                       start: startNode(r).id, 
                       type: type(r), 
                       end: endNode(r).id, 
                       description: r.description
                   }] AS rels
            """)
            
            elapsed_time = time.time() - start_time
            print(f"Collected {len(result)} SLLPA community information, time elapsed: {elapsed_time:.2f} seconds")
            return result
            
        except Exception as e:
            print(f"Error collecting SLLPA community information: {e}")
            return self._collect_info_fallback()
    
    def _collect_info_in_batches(self, total_count: int) -> List[Dict]:
        """Collect community information in batches"""
        batch_size = 50  # Default batch processing size
        if BATCH_SIZE:
            batch_size = min(50, max(10, BATCH_SIZE // 2))  # Adjust to batch size suitable for community collection
            
        total_batches = (total_count + batch_size - 1) // batch_size
        all_results = []
        
        print(f"Using batch processing to collect SLLPA community information, total {total_batches} batches")
        
        for batch in range(total_batches):
            if batch > 20:  # Limit batches
                print(f"Reached maximum batch limit (20), stopping collection")
                break
                
            skip = batch * batch_size
            
            try:
                batch_result = self.graph.query("""
                MATCH (c:`__Community__`)
                WHERE c.level = 0
                WITH c ORDER BY c.community_rank DESC NULLS LAST
                SKIP $skip LIMIT $batch_size
                
                MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
                WITH c, collect(e) as nodes
                WHERE size(nodes) > 1
                
                CALL {
                    WITH nodes
                    WITH nodes[0..20] AS limited_nodes
                    MATCH (n1)-->(n2)
                    WHERE n1 IN limited_nodes AND n2 IN limited_nodes
                    RETURN collect(distinct relationship(n1, n2)) as relationships
                }
                
                RETURN c.id AS communityId,
                       [n in nodes | {
                           id: n.id, 
                           description: n.description, 
                           type: [el in labels(n) WHERE el <> '__Entity__'][0]
                       }] AS nodes,
                       [r in relationships | {
                           start: startNode(r).id, 
                           type: type(r), 
                           end: endNode(r).id, 
                           description: r.description
                       }] AS rels
                """, params={"skip": skip, "batch_size": batch_size})
                
                all_results.extend(batch_result)
                print(f"Batch {batch+1}/{total_batches} completed, collected {len(batch_result)} communities")
                
            except Exception as e:
                print(f"Error processing batch {batch+1}: {e}")
                continue
        
        return all_results
    
    def _collect_info_fallback(self) -> List[Dict]:
        """Fallback information collection method"""
        try:
            print("Trying to collect community information using simplified query...")
            result = self.graph.query("""
            MATCH (c:`__Community__`)
            WHERE c.level = 0
            WITH c LIMIT 50
            MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
            WITH c, collect(e) as nodes
            WHERE size(nodes) > 1
            RETURN c.id AS communityId,
                   [n in nodes | {
                       id: n.id, 
                       description: coalesce(n.description, 'No description'), 
                       type: labels(n)[0]
                   }] AS nodes,
                   [] AS rels
            """)
            
            print(f"Collected {len(result)} community information using simplified query")
            return result
        except Exception as e:
            print(f"Simplified query also failed: {e}")
            return []
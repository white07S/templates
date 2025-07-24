import re
import json
from typing import List, Dict, Any, Optional
import jieba.analyse

from config.neo4jdb import get_db_manager
from search_new.config.search_config import search_config


class CommunityAwareSearchEnhancer:
    """社区感知搜索增强器，利用社区结构优化搜索策略"""
    
    def __init__(self, llm):
        """
        初始化社区感知搜索增强器
        
        参数:
            llm: 大语言模型实例
        """
        self.llm = llm
        self.db_manager = get_db_manager()
        
        # 从配置加载参数
        reasoning_config = search_config.get_reasoning_config()
        self.max_communities = reasoning_config.get("max_communities", 10)
        self.community_relevance_threshold = reasoning_config.get("community_relevance_threshold", 0.5)
    
    def enhance_search_with_community_context(self, query: str, initial_entities: List[str]) -> Dict[str, Any]:
        """
        使用社区上下文增强搜索
        
        参数:
            query: 搜索查询
            initial_entities: 初始实体列表
            
        返回:
            Dict[str, Any]: 增强的搜索上下文
        """
        try:
            # 1. 找到实体所属的社区
            communities = self._find_entity_communities(initial_entities)
            
            # 2. 分析社区相关性
            relevant_communities = self._analyze_community_relevance(query, communities)
            
            # 3. 提取社区知识
            community_knowledge = self._extract_community_knowledge(relevant_communities)
            
            # 4. 生成增强的搜索策略
            search_strategy = self._generate_search_strategy(query, community_knowledge)
            
            return {
                "communities": relevant_communities,
                "community_knowledge": community_knowledge,
                "search_strategy": search_strategy,
                "enhanced_queries": search_strategy.get("follow_up_queries", []),
                "focus_entities": search_strategy.get("focus_entities", [])
            }
            
        except Exception as e:
            print(f"[CommunityAwareSearchEnhancer] 社区增强失败: {e}")
            return {
                "communities": [],
                "community_knowledge": {},
                "search_strategy": {"strategy_type": "basic"},
                "enhanced_queries": [],
                "focus_entities": []
            }
    
    def _find_entity_communities(self, entity_ids: List[str], level: int = 0) -> List[Dict[str, Any]]:
        """
        查找实体所属的社区
        
        参数:
            entity_ids: 实体ID列表
            level: 社区层级
            
        返回:
            List[Dict[str, Any]]: 社区信息列表
        """
        if not entity_ids:
            return []
        
        try:
            cypher = """
            MATCH (e:__Entity__)-[:IN_COMMUNITY]->(c:__Community__)
            WHERE e.id IN $entity_ids AND c.level = $level
            RETURN DISTINCT c.id AS community_id, 
                   c.summary AS summary,
                   c.weight AS weight,
                   c.level AS level,
                   count(e) AS entity_count
            ORDER BY c.weight DESC
            LIMIT $max_communities
            """
            
            result = self.db_manager.execute_query(cypher, {
                "entity_ids": entity_ids,
                "level": level,
                "max_communities": self.max_communities
            })
            
            if not result.empty:
                return result.to_dict('records')
            else:
                return []
                
        except Exception as e:
            print(f"[CommunityAwareSearchEnhancer] 查找实体社区失败: {e}")
            return []
    
    def _analyze_community_relevance(self, query: str, communities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        分析社区与查询的相关性
        
        参数:
            query: 查询字符串
            communities: 社区列表
            
        返回:
            List[Dict[str, Any]]: 按相关性排序的社区列表
        """
        if not communities:
            return []
        
        try:
            # 使用jieba提取查询关键词
            query_keywords = set(jieba.analyse.extract_tags(query, topK=10))
            
            scored_communities = []
            for community in communities:
                summary = community.get("summary", "")
                
                # 计算关键词重叠度
                if summary:
                    summary_keywords = set(jieba.analyse.extract_tags(summary, topK=20))
                    overlap = len(query_keywords & summary_keywords)
                    relevance_score = overlap / max(len(query_keywords), 1)
                else:
                    relevance_score = 0.0
                
                # 结合社区权重
                weight = community.get("weight", 0.0)
                final_score = relevance_score * 0.7 + (weight / 100.0) * 0.3
                
                community_copy = community.copy()
                community_copy["relevance_score"] = final_score
                scored_communities.append(community_copy)
            
            # 按相关性排序并过滤
            scored_communities.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # 只返回相关性超过阈值的社区
            relevant_communities = [
                c for c in scored_communities 
                if c["relevance_score"] >= self.community_relevance_threshold
            ]
            
            return relevant_communities[:self.max_communities]
            
        except Exception as e:
            print(f"[CommunityAwareSearchEnhancer] 分析社区相关性失败: {e}")
            return communities  # 返回原始社区列表
    
    def _extract_community_knowledge(self, communities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        从社区中提取知识
        
        参数:
            communities: 社区列表
            
        返回:
            Dict[str, Any]: 提取的社区知识
        """
        if not communities:
            return {"entities": [], "relationships": [], "summaries": []}
        
        try:
            community_ids = [c["community_id"] for c in communities]
            
            # 获取社区内的实体
            entities_cypher = """
            MATCH (c:__Community__)<-[:IN_COMMUNITY]-(e:__Entity__)
            WHERE c.id IN $community_ids
            RETURN e.id AS entity_id, e.description AS description
            LIMIT 50
            """
            
            entities_result = self.db_manager.execute_query(entities_cypher, {
                "community_ids": community_ids
            })
            
            entities = entities_result.to_dict('records') if not entities_result.empty else []
            
            # 获取社区内的关系
            relationships_cypher = """
            MATCH (c:__Community__)<-[:IN_COMMUNITY]-(e1:__Entity__)
            MATCH (e1)-[r]-(e2:__Entity__)-[:IN_COMMUNITY]->(c)
            WHERE c.id IN $community_ids
            RETURN r.description AS description, r.weight AS weight
            ORDER BY r.weight DESC
            LIMIT 30
            """
            
            relationships_result = self.db_manager.execute_query(relationships_cypher, {
                "community_ids": community_ids
            })
            
            relationships = relationships_result.to_dict('records') if not relationships_result.empty else []
            
            # 提取社区摘要
            summaries = [c.get("summary", "") for c in communities if c.get("summary")]
            
            return {
                "entities": entities,
                "relationships": relationships,
                "summaries": summaries
            }
            
        except Exception as e:
            print(f"[CommunityAwareSearchEnhancer] 提取社区知识失败: {e}")
            return {"entities": [], "relationships": [], "summaries": []}
    
    def _generate_search_strategy(self, query: str, community_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于社区知识生成搜索策略
        
        参数:
            query: 用户查询
            community_knowledge: 社区知识
            
        返回:
            Dict[str, Any]: 搜索策略
        """
        entities = community_knowledge.get("entities", [])
        relationships = community_knowledge.get("relationships", [])
        
        # 如果没有足够的社区信息，返回基本策略
        if len(entities) < 3:
            return {
                "strategy_type": "basic",
                "follow_up_queries": [],
                "focus_entities": []
            }
        
        try:
            # 构建提示
            prompt = f"""
            基于用户查询和社区知识，生成一个最多3个后续搜索查询的列表。
            
            用户查询: {query}
            
            社区中的关键实体:
            {', '.join([e['entity_id'] for e in entities[:10]])}
            
            请考虑这些实体之间的关系，生成更深入的查询以获取全面信息。
            返回JSON格式的后续查询和关注实体。
            """
            
            # 调用LLM生成搜索策略
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 尝试解析JSON响应
            try:
                strategy = json.loads(content)
                if isinstance(strategy, dict):
                    return {
                        "strategy_type": "community_enhanced",
                        "follow_up_queries": strategy.get("follow_up_queries", [])[:3],
                        "focus_entities": strategy.get("focus_entities", [])[:10]
                    }
            except json.JSONDecodeError:
                pass
            
            # 如果JSON解析失败，使用简单的文本解析
            queries = self._extract_queries_from_text(content)
            focus_entities = [e['entity_id'] for e in entities[:5]]
            
            return {
                "strategy_type": "community_enhanced",
                "follow_up_queries": queries[:3],
                "focus_entities": focus_entities
            }
            
        except Exception as e:
            print(f"[CommunityAwareSearchEnhancer] 生成搜索策略失败: {e}")
            return {
                "strategy_type": "basic",
                "follow_up_queries": [],
                "focus_entities": []
            }
    
    def _extract_queries_from_text(self, text: str) -> List[str]:
        """从文本中提取查询"""
        queries = []
        
        # 使用jieba提取关键词
        keywords = jieba.analyse.extract_tags(text, topK=10)
        
        # 从原始内容中提取可能的查询
        query_pattern = r'"([^"]+)"'
        quoted_queries = re.findall(query_pattern, text)
        
        for query in quoted_queries:
            if len(query.strip()) > 5:
                queries.append(query.strip())
        
        # 如果没有找到引号引起的查询，尝试提取句子
        if not queries:
            sentence_pattern = r'[？?!！。；;][^？?!！。；;]{5,50}[？?!！。；;]'
            sentences = re.findall(sentence_pattern, text)
            queries = [s.strip() for s in sentences if len(s.strip()) > 10][:3]
        
        return queries
    
    def get_community_summary(self, community_id: str) -> Dict[str, Any]:
        """
        获取特定社区的详细摘要
        
        参数:
            community_id: 社区ID
            
        返回:
            Dict[str, Any]: 社区摘要信息
        """
        try:
            cypher = """
            MATCH (c:__Community__)
            WHERE c.id = $community_id
            RETURN c.id AS community_id,
                   c.summary AS summary,
                   c.weight AS weight,
                   c.level AS level
            """
            
            result = self.db_manager.execute_query(cypher, {"community_id": community_id})
            
            if not result.empty:
                return result.iloc[0].to_dict()
            else:
                return {}
                
        except Exception as e:
            print(f"[CommunityAwareSearchEnhancer] 获取社区摘要失败: {e}")
            return {}
    
    def find_related_communities(self, entity_ids: List[str], max_hops: int = 2) -> List[Dict[str, Any]]:
        """
        查找与给定实体相关的社区（通过关系扩展）
        
        参数:
            entity_ids: 实体ID列表
            max_hops: 最大跳数
            
        返回:
            List[Dict[str, Any]]: 相关社区列表
        """
        try:
            cypher = f"""
            MATCH (start:__Entity__)-[*1..{max_hops}]-(related:__Entity__)
            WHERE start.id IN $entity_ids
            MATCH (related)-[:IN_COMMUNITY]->(c:__Community__)
            RETURN DISTINCT c.id AS community_id,
                   c.summary AS summary,
                   c.weight AS weight,
                   count(DISTINCT related) AS related_entity_count
            ORDER BY c.weight DESC, related_entity_count DESC
            LIMIT $max_communities
            """
            
            result = self.db_manager.execute_query(cypher, {
                "entity_ids": entity_ids,
                "max_communities": self.max_communities
            })
            
            if not result.empty:
                return result.to_dict('records')
            else:
                return []
                
        except Exception as e:
            print(f"[CommunityAwareSearchEnhancer] 查找相关社区失败: {e}")
            return []

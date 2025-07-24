from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time
import pandas as pd

from langchain_core.tools import BaseTool

from model.get_models import get_llm_model, get_embeddings_model
from CacheManage.manager import CacheManager, ContextAndKeywordAwareCacheKeyStrategy, MemoryCacheBackend
from config.neo4jdb import get_db_manager
from search_new.utils.vector_utils import VectorUtils
from search_new.utils.search_utils import SearchUtils
from search_new.config.search_config import search_config


class BaseSearchTool(ABC):
    """搜索工具基础类，为各种搜索实现提供通用功能"""
    
    def __init__(self, cache_dir: Optional[str] = None, tool_name: str = "search"):
        """
        初始化搜索工具
        
        参数:
            cache_dir: 缓存目录，如果为None则使用配置中的默认目录
            tool_name: 工具名称，用于日志和缓存
        """
        self.tool_name = tool_name
        
        # 初始化大语言模型和嵌入模型
        self.llm = get_llm_model()
        self.embeddings = get_embeddings_model()
        
        # 设置缓存目录
        if cache_dir is None:
            cache_dir = search_config.get_cache_dir("base")
        
        # 初始化缓存管理器
        self.cache_manager = CacheManager(
            key_strategy=ContextAndKeywordAwareCacheKeyStrategy(),
            storage_backend=MemoryCacheBackend(
                max_size=search_config.get_cache_max_size()
            ),
            cache_dir=cache_dir
        )
        
        # 性能监控指标
        self.performance_metrics = {
            "query_time": 0,  # 数据库查询时间
            "llm_time": 0,    # 大语言模型处理时间
            "total_time": 0   # 总处理时间
        }
        
        # 初始化Neo4j连接
        self._setup_neo4j()
        
        # 加载配置
        self._load_config()
    
    def _setup_neo4j(self):
        """设置Neo4j连接"""
        # 获取数据库连接管理器
        db_manager = get_db_manager()
        
        # 获取图数据库实例
        self.graph = db_manager.get_graph()
        
        # 获取驱动（用于直接执行查询）
        self.driver = db_manager.get_driver()
        
        # 获取连接信息
        self.neo4j_uri = db_manager.neo4j_uri
        self.neo4j_username = db_manager.neo4j_username
        self.neo4j_password = db_manager.neo4j_password
    
    def _load_config(self):
        """加载配置参数"""
        # 从配置中加载通用参数
        self.response_type = search_config.get_response_type()
        self.kb_name = search_config.get_kb_name()
        self.max_workers = search_config.get_max_workers()
        self.batch_size = search_config.get_batch_size()
    
    def db_query(self, cypher: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """
        执行Cypher查询
        
        参数:
            cypher: Cypher查询语句
            params: 查询参数
            
        返回:
            pd.DataFrame: 查询结果
        """
        if params is None:
            params = {}
            
        start_time = time.time()
        
        try:
            # 使用连接管理器执行查询
            result = get_db_manager().execute_query(cypher, params)
            
            # 记录查询时间
            self.performance_metrics["query_time"] += time.time() - start_time
            
            return result
            
        except Exception as e:
            print(f"[{self.tool_name}] 数据库查询失败: {e}")
            self.performance_metrics["query_time"] += time.time() - start_time
            return pd.DataFrame()
    
    def vector_search(self, query: str, limit: int = 10) -> List[str]:
        """
        基于向量相似度的搜索方法
        
        参数:
            query: 搜索查询
            limit: 最大返回结果数
            
        返回:
            List[str]: 匹配实体ID列表
        """
        try:
            # 获取查询的嵌入向量
            query_embedding = self.embeddings.embed_query(query)
            
            # 构建向量搜索查询
            cypher = f"""
            CALL db.index.vector.queryNodes('{search_config.get_vector_index_name()}', $limit, $query_embedding)
            YIELD node, score
            RETURN node.id AS id, score
            ORDER BY score DESC
            """
            
            result = self.db_query(cypher, {
                "query_embedding": query_embedding,
                "limit": limit
            })
            
            if not result.empty:
                return result['id'].tolist()
            else:
                return []
                
        except Exception as e:
            print(f"[{self.tool_name}] 向量搜索失败: {e}")
            return []
    
    def text_search(self, query: str, limit: int = 5) -> List[str]:
        """
        基于文本匹配的搜索方法（作为向量搜索的备选）
        
        参数:
            query: 搜索查询
            limit: 最大返回结果数
            
        返回:
            List[str]: 匹配实体ID列表
        """
        try:
            # 构建全文搜索查询
            cypher = """
            MATCH (e:__Entity__)
            WHERE e.id CONTAINS $query OR e.description CONTAINS $query
            RETURN e.id AS id
            LIMIT $limit
            """
            
            result = self.db_query(cypher, {
                "query": query,
                "limit": limit
            })
            
            if not result.empty:
                return result['id'].tolist()
            else:
                return []
                
        except Exception as e:
            print(f"[{self.tool_name}] 文本搜索失败: {e}")
            return []
    
    def get_entity_info(self, entity_ids: List[str]) -> List[Dict[str, Any]]:
        """
        获取实体详细信息
        
        参数:
            entity_ids: 实体ID列表
            
        返回:
            List[Dict[str, Any]]: 实体信息列表
        """
        if not entity_ids:
            return []
        
        try:
            cypher = """
            MATCH (e:__Entity__)
            WHERE e.id IN $entity_ids
            RETURN e.id AS id, e.description AS description
            """
            
            result = self.db_query(cypher, {"entity_ids": entity_ids})
            
            if not result.empty:
                return result.to_dict('records')
            else:
                return []
                
        except Exception as e:
            print(f"[{self.tool_name}] 获取实体信息失败: {e}")
            return []
    
    def get_relationships(self, entity_ids: List[str], max_rels: int = 20) -> List[Dict[str, Any]]:
        """
        获取实体间的关系信息
        
        参数:
            entity_ids: 实体ID列表
            max_rels: 最大关系数量
            
        返回:
            List[Dict[str, Any]]: 关系信息列表
        """
        if not entity_ids:
            return []
        
        try:
            cypher = """
            MATCH (e1:__Entity__)-[r]-(e2:__Entity__)
            WHERE e1.id IN $entity_ids OR e2.id IN $entity_ids
            RETURN r.description AS description, r.weight AS weight
            ORDER BY r.weight DESC
            LIMIT $max_rels
            """
            
            result = self.db_query(cypher, {
                "entity_ids": entity_ids,
                "max_rels": max_rels
            })
            
            if not result.empty:
                return result.to_dict('records')
            else:
                return []
                
        except Exception as e:
            print(f"[{self.tool_name}] 获取关系信息失败: {e}")
            return []
    
    def get_communities(self, entity_ids: List[str], level: int = 0, max_communities: int = 5) -> List[Dict[str, Any]]:
        """
        获取实体所属的社区信息
        
        参数:
            entity_ids: 实体ID列表
            level: 社区层级
            max_communities: 最大社区数量
            
        返回:
            List[Dict[str, Any]]: 社区信息列表
        """
        if not entity_ids:
            return []
        
        try:
            cypher = """
            MATCH (e:__Entity__)-[:IN_COMMUNITY]->(c:__Community__)
            WHERE e.id IN $entity_ids AND c.level = $level
            RETURN DISTINCT c.id AS id, c.summary AS summary, c.weight AS weight
            ORDER BY c.weight DESC
            LIMIT $max_communities
            """
            
            result = self.db_query(cypher, {
                "entity_ids": entity_ids,
                "level": level,
                "max_communities": max_communities
            })
            
            if not result.empty:
                return result.to_dict('records')
            else:
                return []
                
        except Exception as e:
            print(f"[{self.tool_name}] 获取社区信息失败: {e}")
            return []
    
    def get_chunks(self, entity_ids: List[str], max_chunks: int = 10) -> List[Dict[str, Any]]:
        """
        获取与实体相关的文本块
        
        参数:
            entity_ids: 实体ID列表
            max_chunks: 最大文本块数量
            
        返回:
            List[Dict[str, Any]]: 文本块信息列表
        """
        if not entity_ids:
            return []
        
        try:
            cypher = """
            MATCH (e:__Entity__)<-[:MENTIONS]-(c:__Chunk__)
            WHERE e.id IN $entity_ids
            WITH c, count(DISTINCT e) as entity_count
            RETURN c.id AS id, c.text AS text
            ORDER BY entity_count DESC
            LIMIT $max_chunks
            """
            
            result = self.db_query(cypher, {
                "entity_ids": entity_ids,
                "max_chunks": max_chunks
            })
            
            if not result.empty:
                return result.to_dict('records')
            else:
                return []
                
        except Exception as e:
            print(f"[{self.tool_name}] 获取文本块失败: {e}")
            return []
    
    @abstractmethod
    def _setup_chains(self):
        """
        设置处理链，子类必须实现
        用于配置各种LLM处理链和提示模板
        """
        pass
    
    @abstractmethod
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取关键词
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 关键词字典，包含低级和高级关键词
        """
        pass
    
    @abstractmethod
    def search(self, query: Any) -> str:
        """
        执行搜索
        
        参数:
            query: 查询内容，可以是字符串或包含更多信息的字典
            
        返回:
            str: 搜索结果
        """
        pass
    
    def get_tool(self) -> BaseTool:
        """
        获取搜索工具实例
        
        返回:
            BaseTool: 搜索工具
        """
        # 创建动态工具类
        class DynamicSearchTool(BaseTool):
            name: str = f"{self.tool_name}"
            description: str = f"高级搜索工具，用于在知识库中查找信息"
            
            def _run(self_tool, query: Any) -> str:
                return self.search(query)
            
            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")
        
        return DynamicSearchTool()
    
    def _log_performance(self, operation: str, start_time: float):
        """
        记录性能指标
        
        参数:
            operation: 操作名称
            start_time: 开始时间
        """
        duration = time.time() - start_time
        self.performance_metrics[operation] = duration
        print(f"[{self.tool_name}] 性能指标 - {operation}: {duration:.4f}s")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        return self.performance_metrics.copy()
    
    def reset_performance_metrics(self):
        """重置性能指标"""
        self.performance_metrics = {
            "query_time": 0,
            "llm_time": 0,
            "total_time": 0
        }
    
    def close(self):
        """关闭资源连接"""
        # 子类可以重写此方法来清理资源
        pass
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

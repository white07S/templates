from typing import Dict, Any
from config.settings import (
    LOCAL_SEARCH_CONFIG, 
    GLOBAL_SEARCH_CONFIG, 
    SEARCH_CACHE_CONFIG, 
    REASONING_CONFIG,
    MAX_WORKERS,
    BATCH_SIZE,
    EMBEDDING_BATCH_SIZE,
    LLM_BATCH_SIZE,
    response_type,
    KB_NAME
)

class SearchConfig:
    """搜索模块统一配置类"""
    
    def __init__(self):
        """初始化搜索配置"""
        self._load_configs()
    
    def _load_configs(self):
        """从settings.py加载配置"""
        # 本地搜索配置
        self.local_search = LOCAL_SEARCH_CONFIG.copy()
        
        # 全局搜索配置
        self.global_search = GLOBAL_SEARCH_CONFIG.copy()
        
        # 缓存配置
        self.cache = SEARCH_CACHE_CONFIG.copy()
        
        # 推理配置
        self.reasoning = REASONING_CONFIG.copy()
        
        # 性能配置
        self.performance = {
            "max_workers": MAX_WORKERS,
            "batch_size": BATCH_SIZE,
            "embedding_batch_size": EMBEDDING_BATCH_SIZE,
            "llm_batch_size": LLM_BATCH_SIZE
        }
        
        # 通用配置
        self.general = {
            "response_type": response_type,
            "kb_name": KB_NAME
        }
    
    def get_local_search_config(self) -> Dict[str, Any]:
        """获取本地搜索配置"""
        return self.local_search
    
    def get_global_search_config(self) -> Dict[str, Any]:
        """获取全局搜索配置"""
        return self.global_search
    
    def get_cache_config(self) -> Dict[str, Any]:
        """获取缓存配置"""
        return self.cache
    
    def get_reasoning_config(self) -> Dict[str, Any]:
        """获取推理配置"""
        return self.reasoning
    
    def get_performance_config(self) -> Dict[str, Any]:
        """获取性能配置"""
        return self.performance
    
    def get_general_config(self) -> Dict[str, Any]:
        """获取通用配置"""
        return self.general
    
    def get_vector_index_name(self) -> str:
        """获取向量索引名称"""
        return self.local_search.get("index_name", "vector")
    
    def get_retrieval_query(self) -> str:
        """获取检索查询模板"""
        return self.local_search.get("retrieval_query", "")
    
    def get_top_entities(self) -> int:
        """获取最大实体检索数量"""
        return self.local_search.get("top_entities", 10)
    
    def get_top_chunks(self) -> int:
        """获取最大文本块检索数量"""
        return self.local_search.get("top_chunks", 10)
    
    def get_top_communities(self) -> int:
        """获取最大社区检索数量"""
        return self.local_search.get("top_communities", 2)
    
    def get_top_outside_rels(self) -> int:
        """获取最大外部关系检索数量"""
        return self.local_search.get("top_outside_rels", 10)
    
    def get_top_inside_rels(self) -> int:
        """获取最大内部关系检索数量"""
        return self.local_search.get("top_inside_rels", 10)
    
    def get_global_search_level(self) -> int:
        """获取全局搜索默认层级"""
        return self.global_search.get("default_level", 0)
    
    def get_global_batch_size(self) -> int:
        """获取全局搜索批处理大小"""
        return self.global_search.get("batch_size", 10)
    
    def get_max_communities(self) -> int:
        """获取最大社区处理数量"""
        return self.global_search.get("max_communities", 100)
    
    def get_cache_dir(self, cache_type: str = "base") -> str:
        """
        获取缓存目录
        
        参数:
            cache_type: 缓存类型 (base, local_search, global_search, deep_research)
        """
        cache_dirs = {
            "base": self.cache.get("base_cache_dir", "./cache"),
            "local_search": self.cache.get("local_search_cache_dir", "./cache/local_search"),
            "global_search": self.cache.get("global_search_cache_dir", "./cache/global_search"),
            "deep_research": self.cache.get("deep_research_cache_dir", "./cache/deep_research")
        }
        return cache_dirs.get(cache_type, self.cache.get("base_cache_dir", "./cache"))
    
    def get_cache_max_size(self) -> int:
        """获取缓存最大大小"""
        return self.cache.get("max_cache_size", 200)
    
    def get_cache_ttl(self) -> int:
        """获取缓存TTL（秒）"""
        return self.cache.get("cache_ttl", 3600)
    
    def is_memory_cache_enabled(self) -> bool:
        """是否启用内存缓存"""
        return self.cache.get("memory_cache_enabled", True)
    
    def is_disk_cache_enabled(self) -> bool:
        """是否启用磁盘缓存"""
        return self.cache.get("disk_cache_enabled", True)
    
    def get_max_iterations(self) -> int:
        """获取推理最大迭代次数"""
        return self.reasoning.get("max_iterations", 5)
    
    def get_max_search_limit(self) -> int:
        """获取推理最大搜索限制"""
        return self.reasoning.get("max_search_limit", 10)
    
    def get_thinking_depth(self) -> int:
        """获取思考深度"""
        return self.reasoning.get("thinking_depth", 3)
    
    def get_exploration_width(self) -> int:
        """获取探索宽度"""
        return self.reasoning.get("exploration_width", 3)
    
    def get_max_exploration_steps(self) -> int:
        """获取最大探索步数"""
        return self.reasoning.get("max_exploration_steps", 5)
    
    def get_max_evidence_items(self) -> int:
        """获取最大证据项数量"""
        return self.reasoning.get("max_evidence_items", 50)
    
    def get_evidence_relevance_threshold(self) -> float:
        """获取证据相关性阈值"""
        return self.reasoning.get("evidence_relevance_threshold", 0.7)
    
    def get_validation_config(self) -> Dict[str, Any]:
        """获取验证配置"""
        return self.reasoning.get("validation", {})
    
    def get_exploration_config(self) -> Dict[str, Any]:
        """获取探索配置"""
        return self.reasoning.get("exploration", {})
    
    def get_response_type(self) -> str:
        """获取响应类型"""
        return self.general.get("response_type", "多个段落")
    
    def get_kb_name(self) -> str:
        """获取知识库名称"""
        return self.general.get("kb_name", "")
    
    def get_max_workers(self) -> int:
        """获取最大工作线程数"""
        return self.performance.get("max_workers", 4)
    
    def get_batch_size(self) -> int:
        """获取批处理大小"""
        return self.performance.get("batch_size", 100)
    
    def get_embedding_batch_size(self) -> int:
        """获取嵌入向量批处理大小"""
        return self.performance.get("embedding_batch_size", 64)
    
    def get_llm_batch_size(self) -> int:
        """获取LLM批处理大小"""
        return self.performance.get("llm_batch_size", 5)
    
    def update_config(self, section: str, key: str, value: Any):
        """
        更新配置项
        
        参数:
            section: 配置段 (local_search, global_search, cache, reasoning, performance, general)
            key: 配置键
            value: 配置值
        """
        if hasattr(self, section):
            config_section = getattr(self, section)
            if isinstance(config_section, dict):
                config_section[key] = value
            else:
                setattr(self, f"{section}_{key}", value)
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            "local_search": self.local_search,
            "global_search": self.global_search,
            "cache": self.cache,
            "reasoning": self.reasoning,
            "performance": self.performance,
            "general": self.general
        }

# 全局配置实例
search_config = SearchConfig()

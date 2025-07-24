import time
import json
from typing import List, Dict, Any

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.prompt import MAP_SYSTEM_PROMPT, REDUCE_SYSTEM_PROMPT
from config.settings import gl_description
from search_new.tools.base import BaseSearchTool
from search_new.core.global_search import GlobalSearch
from search_new.config.search_config import search_config


class GlobalSearchTool(BaseSearchTool):
    """全局搜索工具，基于知识图谱和Map-Reduce模式实现跨社区的广泛查询"""

    def __init__(self, level: int = None):
        """
        初始化全局搜索工具
        
        参数:
            level: 社区层级，如果为None则使用配置中的默认值
        """
        # 设置社区层级
        self.level = level if level is not None else search_config.get_global_search_level()
        
        # 调用父类构造函数
        super().__init__(cache_dir=search_config.get_cache_dir("global_search"))

        # 初始化全局搜索核心类
        self.global_searcher = GlobalSearch(self.llm, self.response_type)

        # 设置处理链
        self._setup_chains()

    def _setup_chains(self):
        """设置处理链"""
        # 创建Map阶段处理链
        self.map_prompt = ChatPromptTemplate.from_messages([
            ("system", MAP_SYSTEM_PROMPT),
            ("human", """
                ---数据表格--- 
                {context_data}
                
                用户的问题是：
                {question}
                """),
        ])
        
        # 创建Reduce阶段处理链
        self.reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", REDUCE_SYSTEM_PROMPT),
            ("human", """
                ---中间分析结果--- 
                {context_data}
                
                用户的问题是：
                {question}
                
                请基于以上分析结果，生成一个综合性的回答。
                """),
        ])
        
        # 链接到LLM
        self.map_chain = self.map_prompt | self.llm | StrOutputParser()
        self.reduce_chain = self.reduce_prompt | self.llm | StrOutputParser()

    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取关键词（全局搜索工具不需要复杂的关键词提取）
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 关键词字典
        """
        # 简单的关键词提取，主要用于缓存
        from search_new.utils.search_utils import SearchUtils
        simple_keywords = SearchUtils.extract_keywords_from_text(query, max_keywords=5)
        return {
            "low_level": simple_keywords[:3],
            "high_level": simple_keywords[3:],
            "keywords": simple_keywords
        }
    
    def search(self, query_input: Any) -> List[str]:
        """
        执行全局搜索，实现Map-Reduce模式
        
        参数:
            query_input: 查询输入，可以是字符串或包含查询和关键词的字典
            
        返回:
            List[str]: 中间结果列表（用于GraphAgent的reduce阶段）
        """
        overall_start = time.time()
        
        # 解析输入
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
            keywords = query_input.get("keywords", [])
        else:
            query = str(query_input)
            # 提取关键词
            extracted_keywords = self.extract_keywords(query)
            keywords = extracted_keywords.get("keywords", [])
        
        # 检查缓存
        cache_key = query
        if keywords:
            cache_key = f"{query}||{','.join(sorted(keywords))}"
        
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # 使用核心全局搜索的Map阶段
            results = self.global_searcher.search_with_map_only(query, level=self.level)
            
            # 缓存结果
            self.cache_manager.set(cache_key, results)
            
            # 记录性能指标
            self.performance_metrics["total_time"] = time.time() - overall_start

            return results
            
        except Exception as e:
            print(f"[{self.tool_name}] 全局搜索失败: {e}")
            
            # 记录性能指标
            self.performance_metrics["total_time"] = time.time() - overall_start
            
            return []
    
    def get_tool(self) -> BaseTool:
        """
        获取搜索工具
        
        返回:
            BaseTool: 搜索工具实例
        """
        class GlobalRetrievalTool(BaseTool):
            name : str= "global_retriever"
            description : str = gl_description
            
            def _run(self_tool, query: Any) -> List[str]:
                return self.search(query)
            
            def _arun(self_tool, query: Any) -> List[str]:
                raise NotImplementedError("异步执行未实现")
        
        return GlobalRetrievalTool()
    
    def close(self):
        """关闭资源"""
        if hasattr(self, 'global_searcher'):
            self.global_searcher.close()
        super().close()

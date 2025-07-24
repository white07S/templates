import time
import json
from typing import Dict, List, Any

from langchain_core.tools import BaseTool, create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langsmith import traceable

from config.prompt import LC_SYSTEM_PROMPT
from config.settings import lc_description
from search_new.tools.base import BaseSearchTool
from search_new.core.local_search import LocalSearch
from search_new.config.search_config import search_config


class LocalSearchTool(BaseSearchTool):
    """本地搜索工具，基于向量检索实现社区内部的精确查询"""
    
    def __init__(self):
        """初始化本地搜索工具"""
        # 调用父类构造函数
        super().__init__(cache_dir=search_config.get_cache_dir("local_search"))
        
        # 设置聊天历史，用于连续对话
        self.chat_history = []
                
        # 创建本地搜索器和检索器
        self.local_searcher = LocalSearch(self.llm, self.embeddings)
        self.retriever = self.local_searcher.as_retriever()

        # 设置处理链
        self._setup_chains()

    def _setup_chains(self):
        """设置处理链"""
        # 创建RAG提示模板
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            ("human", """
                ---分析报告--- 
                请注意，下面提供的分析报告按**重要性降序排列**。

                {context}

                用户的问题是：
                {input}

                请按以下格式输出回答：
                1. 使用三级标题(###)标记主题
                2. 主要内容用清晰的段落展示
                3. 最后必须用"#### 引用数据"标记引用部分，列出用到的数据来源
                """
             )
        ])
        
        # 创建RAG链
        self.rag_chain = (
            {
                "context": self.retriever | self._format_docs,
                "input": RunnablePassthrough(),
                "response_type": lambda _: self.response_type
            }
            | rag_prompt
            | self.llm
            | StrOutputParser()
            | self._format_response
        )

    def _format_docs(self, docs):
        """格式化检索到的文档"""
        if not docs:
            return ""
        return docs[0].page_content if docs else ""

    def _format_response(self, response: str) -> Dict[str, Any]:
        """格式化响应结果"""
        return {
            "answer": response,
            "source": "local_search"
        }

    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取关键词（本地搜索工具不需要复杂的关键词提取）
        
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
        }

    @traceable
    def search(self, query_input: Any) -> str:
        """
        执行本地搜索
        
        参数:
            query_input: 查询输入，可以是字符串或字典
            
        返回:
            str: 搜索结果
        """
        overall_start = time.time()
        
        # 解析输入
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
            keywords = query_input.get("keywords", [])
        else:
            query = str(query_input)
            keywords = []
        
        # 检查缓存
        cache_key = query
        if keywords:
            cache_key = f"{query}||{','.join(sorted(keywords))}"
        
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        # 直接使用LocalSearch执行搜索
        try:
            result = self.local_searcher.search(query)
            
            # 缓存结果
            self.cache_manager.set(cache_key, result)
            
            # 记录性能指标
            self.performance_metrics["total_time"] = time.time() - overall_start

            if not result:
                return "未找到相关信息"
            return result
        except Exception as e:
            print(f"[{self.tool_name}] 本地搜索失败: {e}")
            error_msg = f"搜索过程中出现问题: {str(e)}"
            
            # 记录性能指标
            self.performance_metrics["total_time"] = time.time() - overall_start
            
            return error_msg
    
    def get_tool(self):
        """
        返回LangChain兼容的检索工具
        
        返回:
            BaseTool: 检索工具实例
        """
        return create_retriever_tool(
            self.retriever,
            "lc_search_tool",
            lc_description,
        )

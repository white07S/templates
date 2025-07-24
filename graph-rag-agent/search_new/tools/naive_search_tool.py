from typing import List, Dict, Any
import time
import numpy as np

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.prompt import NAIVE_PROMPT
from config.settings import response_type, naive_description
from search_new.tools.base import BaseSearchTool
from search_new.utils.vector_utils import VectorUtils
from search_new.config.search_config import search_config


class NaiveSearchTool(BaseSearchTool):
    """简单的Naive RAG搜索工具，只使用embedding进行向量搜索"""
    
    def __init__(self):
        """初始化Naive搜索工具"""
        # 调用父类构造函数
        super().__init__(cache_dir=search_config.get_cache_dir("base"))
        
        # 搜索参数设置
        self.top_k = 3 # 检索的最大文档数量
        
        # 设置处理链
        self._setup_chains()
        
    def _setup_chains(self):
        """设置处理链"""
        # 创建查询处理链
        self.query_prompt = ChatPromptTemplate.from_messages([
            ("system", NAIVE_PROMPT),
            ("human", """
                ---文档片段--- 
                {context}
                
                问题：
                {query}
                """),
        ])
        
        # 链接到LLM
        self.query_chain = self.query_prompt | self.llm | StrOutputParser()
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取关键词（naive rag不需要复杂的关键词提取）
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 空的关键词字典
        """
        return {"low_level": [], "high_level": []}
    
    def search(self, query_input: Any) -> str:
        """
        执行简单的向量搜索
        
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
        
        try:
            # 生成查询的嵌入向量
            search_start = time.time()
            query_embedding = self.embeddings.embed_query(query)
            
            # 获取带embedding的Chunk节点
            chunks_with_embedding = self.graph.query("""
            MATCH (c:__Chunk__)
            WHERE c.embedding IS NOT NULL
            RETURN c.id AS id, c.text AS text, c.embedding AS embedding
            LIMIT 100  // 获取候选集
            """)
            
            # 使用工具类对候选集进行排序
            scored_chunks = VectorUtils.rank_by_similarity(
                query_embedding,
                chunks_with_embedding,
                "embedding",
                self.top_k
            )
            
            # 取top_k个结果
            results = scored_chunks[:self.top_k]
            
            search_time = time.time() - search_start
            self.performance_metrics["query_time"] = search_time
            
            if not results:
                return f"没有找到与'{query}'相关的信息。\n\n{{'data': {{'Chunks':[] }} }}"
            
            # 构建上下文
            context_parts = []
            for i, chunk in enumerate(results, 1):
                text = chunk.get('text', '')
                score = chunk.get('score', 0)
                context_parts.append(f"文档{i} (相似度: {score:.3f}):\n{text}")
            
            context = "\n\n".join(context_parts)
            
            # 使用LLM生成答案
            llm_start = time.time()
            
            answer = self.query_chain.invoke({
                "context": context,
                "query": query,
                "response_type": response_type
            })
            
            llm_time = time.time() - llm_start
            self.performance_metrics["llm_time"] = llm_time
            
            # 在答案末尾添加数据格式
            answer += f"\n\n{{'data': {{'Chunks': {results} }} }}"
            
            # 缓存结果
            self.cache_manager.set(cache_key, answer)
            
            # 记录总耗时
            total_time = time.time() - overall_start
            self.performance_metrics["total_time"] = total_time
            
            return answer
            
        except Exception as e:
            error_msg = f"搜索过程中出现错误: {str(e)}"
            print(error_msg)
            return f"搜索过程中出错: {str(e)}\n\n{{'data': {{'Chunks':[] }} }}"
    
    def get_tool(self) -> BaseTool:
        """
        获取搜索工具
        
        返回:
            BaseTool: 搜索工具实例
        """
        class NaiveRetrievalTool(BaseTool):
            name : str= "naive_retriever"
            description : str = naive_description
            
            def _run(self_tool, query: Any) -> str:
                return self.search(query)
            
            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")
        
        return NaiveRetrievalTool()
    
    def close(self):
        """关闭资源"""
        super().close()

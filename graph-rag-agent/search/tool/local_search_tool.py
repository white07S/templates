from typing import List, Dict, Any
import time
import json
from langsmith import traceable
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.tools.retriever import create_retriever_tool
from langchain_core.output_parsers import StrOutputParser

from config.prompt import LC_SYSTEM_PROMPT, contextualize_q_system_prompt
from config.settings import lc_description
from search.tool.base import BaseSearchTool
from search.local_search import LocalSearch


class LocalSearchTool(BaseSearchTool):
    """本地搜索工具，基于向量检索实现社区内部的精确查询"""
    
    def __init__(self):
        """初始化本地搜索工具"""
        # 调用父类构造函数
        super().__init__(cache_dir="./cache/local_search")
        
        # 设置聊天历史，用于连续对话
        self.chat_history = []
                
        # 创建本地搜索器和检索器
        self.local_searcher = LocalSearch(self.llm, self.embeddings)
        self.retriever = self.local_searcher.as_retriever()

        # 设置处理链
        self._setup_chains()
    
    def _setup_chains(self):
        """设置处理链"""
        # 创建上下文理解提示模板
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # 创建历史感知检索器
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm,
            self.retriever,
            contextualize_q_prompt,
        )

        # 创建带历史的本地查询提示模板
        lc_prompt_with_history = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", """
            ---分析报告--- 
            请注意，下面提供的分析报告按**重要性降序排列**。
            
            {context}
            
            用户的问题是：
            {input}

            请使用三级标题(###)标记主题
            """),
        ])

        # 创建问答链
        self.question_answer_chain = create_stuff_documents_chain(
            self.llm,
            lc_prompt_with_history,
        )

        # 创建完整的RAG链
        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever,
            self.question_answer_chain,
        )
        
        # 创建关键词提取链
        self.keyword_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专门从用户查询中提取搜索关键词的助手。你需要将关键词分为两类：
                1. 低级关键词：具体实体名称、人物、地点、具体事件等
                2. 高级关键词：主题、概念、关系类型等
                
                返回格式必须是JSON格式：
                {{
                    "low_level": ["关键词1", "关键词2", ...], 
                    "high_level": ["关键词1", "关键词2", ...]
                }}
                
                注意：
                - 每类提取3-5个关键词即可
                - 不要添加任何解释或其他文本，只返回JSON
                - 如果某类无关键词，则返回空列表
                """),
            ("human", "{query}")
        ])
        
        self.keyword_chain = self.keyword_prompt | self.llm | StrOutputParser()
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取关键词
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 分类关键词字典
        """
        # 检查缓存
        cached_keywords = self.cache_manager.get(f"keywords:{query}")
        if cached_keywords:
            return cached_keywords
            
        try:
            llm_start = time.time()
            
            # 调用LLM提取关键词
            result = self.keyword_chain.invoke({"query": query})
            
            # 解析JSON结果
            keywords = json.loads(result)
            
            # 记录LLM处理时间
            self.performance_metrics["llm_time"] = time.time() - llm_start
            
            # 确保包含必要的键
            if not isinstance(keywords, dict):
                keywords = {}
            if "low_level" not in keywords:
                keywords["low_level"] = []
            if "high_level" not in keywords:
                keywords["high_level"] = []
                
            # 缓存结果
            self.cache_manager.set(f"keywords:{query}", keywords)
            
            return keywords
            
        except Exception as e:
            print(f"关键词提取失败: {e}")
            # 返回空字典作为默认值
            return {"low_level": [], "high_level": []}
    
    def _filter_documents_by_relevance(self, docs, query: str) -> List:
        """
        根据相关性过滤文档
        
        参数:
            docs: 文档列表
            query: 查询字符串
            
        返回:
            List: 按相关性排序的文档列表
        """
        # 使用基类的标准方法
        return self.filter_by_relevance(query, docs, top_k=5)

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
        
        # 使用RAG链执行搜索
        try:
            ai_msg = self.rag_chain.invoke({
                "input": query,
                "response_type": "多个段落",
                "chat_history": self.chat_history,
            })
            
            # 获取结果
            result = ai_msg.get("answer", "抱歉，我无法回答这个问题。")
            
            # 缓存结果
            self.cache_manager.set(cache_key, result)
            
            # 记录性能指标
            self.performance_metrics["total_time"] = time.time() - overall_start

            if not result:
                return "未找到相关信息"
            return result
        except Exception as e:
            print(f"本地搜索失败: {e}")
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
    
    def close(self):
        """关闭资源"""
        # 先调用父类方法关闭基础资源
        super().close()
        
        # 关闭本地搜索器
        if hasattr(self, 'local_searcher'):
            self.local_searcher.close()
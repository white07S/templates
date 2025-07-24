import time
import json
from typing import List, Dict, Any
import pandas as pd
from neo4j import Result

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.prompt import LC_SYSTEM_PROMPT
from config.settings import gl_description, response_type
from search_new.tools.base import BaseSearchTool
from search_new.config.search_config import search_config


class HybridSearchTool(BaseSearchTool):
    """
    混合搜索工具，实现类似LightRAG的双级检索策略
    结合了局部细节检索和全局主题检索
    """
    
    def __init__(self):
        """初始化混合搜索工具"""
        # 检索参数
        self.entity_limit = 15        # 最大检索实体数量
        self.max_hop_distance = 2     # 最大跳数（关系扩展）
        self.top_communities = 3      # 检索社区数量
        self.batch_size = 10          # 批处理大小
        self.community_level = 0      # 默认社区等级
        
        # 调用父类构造函数
        super().__init__(cache_dir=search_config.get_cache_dir("base"))

        # 设置处理链
        self._setup_chains()
    
    def _setup_chains(self):
        """设置处理链"""
        # 创建主查询处理链 - 用于生成最终答案
        self.query_prompt = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            ("human", """
                ---分析报告--- 
                请注意，以下内容组合了低级详细信息和高级主题概念。

                ## 低级内容（实体详细信息）:
                {low_level}
                
                ## 高级内容（主题和概念）:
                {high_level}

                用户的问题是：
                {query}
                
                请综合利用上述信息回答问题，确保回答全面且有深度。
                回答格式应包含：
                1. 主要内容（使用清晰的段落展示）
                2. 在末尾标明引用的数据来源
                """
            )
        ])
        
        # 链接到LLM
        self.query_chain = self.query_prompt | self.llm | StrOutputParser()
        
        # 关键词提取链
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
        从查询中提取双级关键词
        
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
            
            print(f"DEBUG - LLM关键词结果: {result[:100]}...") if len(str(result)) > 100 else print(f"DEBUG - LLM关键词结果: {result}")
            
            # 解析JSON结果
            try:
                # 尝试直接解析
                if isinstance(result, dict):
                    # 结果已经是字典，无需解析
                    keywords = result
                elif isinstance(result, str):
                    # 清理字符串，移除可能导致解析失败的字符
                    result = result.strip()
                    # 检查字符串是否以JSON格式开始
                    if result.startswith('{') and result.endswith('}'):
                        keywords = json.loads(result)
                    else:
                        # 尝试提取JSON部分 - 寻找第一个{和最后一个}
                        start_idx = result.find('{')
                        end_idx = result.rfind('}')
                        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                            json_str = result[start_idx:end_idx+1]
                            keywords = json.loads(json_str)
                        else:
                            # 没有有效的JSON结构，使用简单的关键词提取
                            raise ValueError("No valid JSON structure found")
                else:
                    # 不是字符串也不是字典
                    raise TypeError(f"Unexpected result type: {type(result)}")
                    
            except (json.JSONDecodeError, ValueError, TypeError) as json_err:
                print(f"JSON解析失败: {json_err}，尝试备用方法提取关键词")
                
                # 备用方法：手动提取关键词
                if isinstance(result, str):
                    # 简单分词提取关键词
                    import re
                    # 移除标点符号，按空格分词
                    words = re.findall(r'\b\w+\b', query.lower())
                    # 过滤停用词（简化版，实际需要更完整的停用词表）
                    stopwords = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
                                "in", "on", "at", "to", "for", "with", "by", "about", "of", "and", "or"}
                    keywords = {
                        "high_level": [word for word in words if len(word) > 5 and word not in stopwords][:3],
                        "low_level": [word for word in words if 2 < len(word) <= 5 and word not in stopwords][:3]
                    }
                else:
                    # 完全失败，返回空关键词
                    keywords = {"low_level": [], "high_level": []}
            
            # 记录LLM处理时间
            self.performance_metrics["llm_time"] += time.time() - llm_start
            
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
            # 返回空关键词
            return {"low_level": [], "high_level": []}

    def _retrieve_low_level_content(self, query: str, keywords: List[str]) -> str:
        """
        检索低级内容（实体和关系）

        参数:
            query: 查询字符串
            keywords: 低级关键词列表

        返回:
            str: 格式化的低级内容
        """
        try:
            # 基于关键词搜索实体
            entity_ids = []

            # 使用向量搜索
            if keywords:
                for keyword in keywords[:5]:  # 限制关键词数量
                    ids = self.vector_search(keyword, limit=3)
                    entity_ids.extend(ids)

            # 如果没有找到实体，使用查询本身搜索
            if not entity_ids:
                entity_ids = self.vector_search(query, limit=self.entity_limit)

            # 去重并限制数量
            entity_ids = list(set(entity_ids))[:self.entity_limit]

            if not entity_ids:
                return "未找到相关实体信息。"

            # 获取实体信息
            entities = self.get_entity_info(entity_ids)

            # 获取关系信息
            relationships = self.get_relationships(entity_ids, max_rels=20)

            # 获取相关文本块
            chunks = self.get_chunks(entity_ids, max_chunks=5)

            # 格式化内容
            content_parts = []

            if entities:
                content_parts.append("### 相关实体")
                for entity in entities:
                    content_parts.append(f"- {entity.get('id', '')}: {entity.get('description', '')}")

            if relationships:
                content_parts.append("\n### 相关关系")
                for rel in relationships:
                    content_parts.append(f"- {rel.get('description', '')}")

            if chunks:
                content_parts.append("\n### 相关文档")
                for chunk in chunks:
                    content_parts.append(f"- {chunk.get('text', '')[:200]}...")

            return "\n".join(content_parts) if content_parts else "未找到相关的低级内容。"

        except Exception as e:
            print(f"[{self.tool_name}] 检索低级内容失败: {e}")
            return "检索低级内容时出现错误。"

    def _retrieve_high_level_content(self, query: str, keywords: List[str]) -> str:
        """
        检索高级内容（社区和主题）

        参数:
            query: 查询字符串
            keywords: 高级关键词列表

        返回:
            str: 格式化的高级内容
        """
        try:
            # 基于关键词搜索实体，然后找到相关社区
            entity_ids = []

            if keywords:
                for keyword in keywords[:3]:  # 限制关键词数量
                    ids = self.vector_search(keyword, limit=5)
                    entity_ids.extend(ids)

            # 如果没有找到实体，使用查询本身搜索
            if not entity_ids:
                entity_ids = self.vector_search(query, limit=10)

            # 去重
            entity_ids = list(set(entity_ids))

            if not entity_ids:
                return "未找到相关社区信息。"

            # 获取社区信息
            communities = self.get_communities(
                entity_ids,
                level=self.community_level,
                max_communities=self.top_communities
            )

            # 格式化社区内容
            content_parts = []

            if communities:
                content_parts.append("### 相关社区主题")
                for community in communities:
                    summary = community.get('summary', '')
                    if summary:
                        content_parts.append(f"- {summary}")

            return "\n".join(content_parts) if content_parts else "未找到相关的高级内容。"

        except Exception as e:
            print(f"[{self.tool_name}] 检索高级内容失败: {e}")
            return "检索高级内容时出现错误。"

    def search(self, query_input: Any) -> str:
        """
        执行混合搜索，结合低级和高级内容

        参数:
            query_input: 字符串查询或包含查询和关键词的字典

        返回:
            str: 生成的最终答案
        """
        overall_start = time.time()

        # 解析输入
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
            low_keywords = query_input.get("low_level_keywords", [])
            high_keywords = query_input.get("high_level_keywords", [])
        else:
            query = str(query_input)
            # 提取关键词
            keywords = self.extract_keywords(query)
            low_keywords = keywords.get("low_level", [])
            high_keywords = keywords.get("high_level", [])

        # 检查缓存
        cache_key = query
        if low_keywords or high_keywords:
            cache_key = f"{query}||low:{','.join(sorted(low_keywords))}||high:{','.join(sorted(high_keywords))}"

        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result

        try:
            # 1. 检索低级内容（实体和关系）
            low_level_content = self._retrieve_low_level_content(query, low_keywords)

            # 2. 检索高级内容（社区和主题）
            high_level_content = self._retrieve_high_level_content(query, high_keywords)

            # 3. 生成最终答案
            llm_start = time.time()

            # 调用LLM生成最终答案
            result = self.query_chain.invoke({
                "query": query,
                "low_level": low_level_content,
                "high_level": high_level_content,
                "response_type": response_type
            })

            self.performance_metrics["llm_time"] += time.time() - llm_start

            # 缓存结果
            self.cache_manager.set(cache_key, result)

            self.performance_metrics["total_time"] = time.time() - overall_start

            if not result:
                return "未找到相关信息"
            return result

        except Exception as e:
            print(f"[{self.tool_name}] 混合搜索失败: {e}")
            error_msg = f"搜索过程中出现问题: {str(e)}"

            # 记录性能指标
            self.performance_metrics["total_time"] = time.time() - overall_start

            return error_msg

    def get_local_tool(self) -> BaseTool:
        """
        获取本地搜索工具实例（仅使用低级内容）

        返回:
            BaseTool: 本地搜索工具实例
        """
        class LocalSearchTool(BaseTool):
            name : str = "local_retriever"
            description : str= "用于需要具体细节的查询。检索实体、关系等详细内容。"

            def _run(self_tool, query: Any) -> str:
                # 设置为仅使用低级内容
                if isinstance(query, dict) and "query" in query:
                    original_query = query["query"]
                    keywords = query.get("keywords", [])
                    # 转换为低级关键词
                    low_keywords = keywords
                    query = {
                        "query": original_query,
                        "low_level_keywords": low_keywords,
                        "high_level_keywords": []  # 不使用高级关键词
                    }
                else:
                    # 提取关键词
                    keywords = self.extract_keywords(str(query))
                    query = {
                        "query": str(query),
                        "low_level_keywords": keywords.get("low_level", []),
                        "high_level_keywords": []
                    }

                return self.search(query)

            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")

        return LocalSearchTool()

    def get_global_tool(self) -> BaseTool:
        """
        获取全局搜索工具实例（仅使用高级内容）

        返回:
            BaseTool: 全局搜索工具实例
        """
        class GlobalSearchTool(BaseTool):
            name : str = "global_retriever"
            description : str= gl_description

            def _run(self_tool, query: Any) -> str:
                # 设置为仅使用高级内容
                if isinstance(query, dict) and "query" in query:
                    original_query = query["query"]
                    keywords = query.get("keywords", [])
                    # 转换为高级关键词
                    high_keywords = keywords
                    query = {
                        "query": original_query,
                        "high_level_keywords": high_keywords,
                        "low_level_keywords": []  # 不使用低级关键词
                    }
                else:
                    # 提取关键词
                    keywords = self.extract_keywords(str(query))
                    query = {
                        "query": str(query),
                        "high_level_keywords": keywords.get("high_level", []),
                        "low_level_keywords": []
                    }

                return self.search(query)

            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")

        return GlobalSearchTool()

    def get_tool(self) -> BaseTool:
        """
        获取混合搜索工具实例

        返回:
            BaseTool: 混合搜索工具实例
        """
        class HybridSearchLangChainTool(BaseTool):
            name: str = "hybrid_search"
            description: str = "混合搜索工具，结合局部细节和全局主题进行综合查询"

            def _run(self_tool, query: Any) -> str:
                return self.search(query)

            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")

        return HybridSearchLangChainTool()

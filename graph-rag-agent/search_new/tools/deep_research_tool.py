from typing import Dict, List, Any, Optional, AsyncGenerator
import time
import re
import logging
import json
import traceback
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage
import asyncio

from search_new.tools.base import BaseSearchTool
from search_new.tools.hybrid_tool import HybridSearchTool
from search_new.tools.local_tool import LocalSearchTool
from search_new.tools.global_tool import GlobalSearchTool
from config.reasoning_prompts import BEGIN_SEARCH_QUERY, BEGIN_SEARCH_RESULT, END_SEARCH_RESULT, MAX_SEARCH_LIMIT, \
    END_SEARCH_QUERY, RELEVANT_EXTRACTION_PROMPT, SUB_QUERY_PROMPT, FOLLOWUP_QUERY_PROMPT, FINAL_ANSWER_PROMPT
from search_new.reasoning.nlp.text_processor import extract_between
from search_new.reasoning.prompts.prompt_manager import kb_prompt
from search_new.reasoning.engines.thinking_engine import ThinkingEngine
from search_new.reasoning.engines.validator import AnswerValidator
from search_new.reasoning.engines.search_engine import DualPathSearcher, QueryGenerator
from search_new.config.search_config import search_config
from config.settings import KB_NAME


class DeepResearchTool(BaseSearchTool):
    """
    深度研究工具：整合多种搜索策略，实现多步骤的思考-搜索-推理过程
    
    该工具实现了多步骤的研究过程，可以执行以下步骤：
    1. 思考分析用户问题
    2. 生成搜索查询
    3. 执行搜索
    4. 整合信息并进一步思考
    5. 迭代上述过程直到获得完整答案
    """
    
    def __init__(self):
        """初始化深度研究工具"""
        super().__init__(cache_dir=search_config.get_cache_dir("deep_research"))

        # 关键词缓存
        self._keywords_cache = {}
        
        # 初始化各种工具，用于不同阶段的搜索
        self.hybrid_tool = HybridSearchTool()  # 用于关键词提取和混合搜索
        self.global_tool = GlobalSearchTool()  # 用于社区检索
        self.local_tool = LocalSearchTool()    # 用于本地搜索
        
        # 初始化思考引擎
        self.thinking_engine = ThinkingEngine(self.llm)
        
        # 初始化答案验证器
        self.answer_validator = AnswerValidator(self.llm)
        
        # 初始化双路径搜索器
        self.dual_path_searcher = DualPathSearcher(self.llm)
        
        # 初始化查询生成器
        self.query_generator = QueryGenerator(self.llm)
        
        # 从配置加载参数
        reasoning_config = search_config.get_reasoning_config()
        self.max_iterations = reasoning_config.get("max_iterations", 5)
        self.max_search_limit = reasoning_config.get("max_search_limit", 10)
        
        # 搜索计数器
        self.search_count = 0
        
        # 工具名称
        self.tool_name = "DeepResearchTool"

    def _setup_chains(self):
        """设置处理链（实现抽象方法）"""
        # DeepResearchTool不需要单独的处理链，因为它使用其他工具
        pass
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取关键词
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 包含low_level和high_level关键词的字典
        """
        # 使用缓存
        if query in self._keywords_cache:
            return self._keywords_cache[query]
        
        # 使用混合工具提取关键词
        keywords = self.hybrid_tool.extract_keywords(query)
        
        # 缓存结果
        self._keywords_cache[query] = keywords
        
        return keywords
    
    def search(self, query_input: Any) -> str:
        """
        执行深度研究搜索
        
        参数:
            query_input: 查询输入，可以是字符串或字典
            
        返回:
            str: 研究结果
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
        cache_key = f"deep_research_{query}"
        if keywords:
            cache_key = f"deep_research_{query}||{','.join(sorted(keywords))}"
        
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # 重置搜索计数器
            self.search_count = 0
            
            # 初始化思考引擎
            self.thinking_engine.initialize_with_query(query)
            
            # 执行多步骤研究过程
            research_result = self._conduct_deep_research(query)
            
            # 缓存结果
            self.cache_manager.set(cache_key, research_result)
            
            # 记录性能指标
            self.performance_metrics["total_time"] = time.time() - overall_start
            self.performance_metrics["search_count"] = self.search_count
            
            return research_result
            
        except Exception as e:
            print(f"[{self.tool_name}] 深度研究失败: {e}")
            error_msg = f"深度研究过程中出现问题: {str(e)}"
            
            # 记录性能指标
            self.performance_metrics["total_time"] = time.time() - overall_start
            self.performance_metrics["error"] = str(e)
            
            return error_msg
    
    def _conduct_deep_research(self, query: str) -> str:
        """
        执行深度研究过程
        
        参数:
            query: 用户查询
            
        返回:
            str: 研究结果
        """
        # 收集的信息
        collected_info = []
        
        # 迭代研究过程
        for iteration in range(self.max_iterations):
            print(f"[{self.tool_name}] 开始第 {iteration + 1} 轮研究...")
            
            # 生成下一步查询
            next_query_result = self.thinking_engine.generate_next_query()
            
            if next_query_result["status"] == "max_iterations_reached":
                print(f"[{self.tool_name}] 达到最大迭代次数，结束研究")
                break
            elif next_query_result["status"] == "answer_ready":
                print(f"[{self.tool_name}] 思考引擎认为已有足够信息")
                break
            elif next_query_result["status"] == "error":
                print(f"[{self.tool_name}] 生成查询时出错: {next_query_result['content']}")
                break
            
            # 获取查询列表
            queries = next_query_result.get("queries", [])
            if not queries:
                print(f"[{self.tool_name}] 没有生成新的查询，结束研究")
                break
            
            # 执行搜索
            for search_query in queries:
                if self.search_count >= self.max_search_limit:
                    print(f"[{self.tool_name}] 达到最大搜索次数限制")
                    break
                
                # 执行双路径搜索
                search_result = self.dual_path_searcher.search(search_query)
                self.search_count += 1
                
                # 记录执行的查询
                self.thinking_engine.add_executed_query(search_query)
                
                # 添加搜索结果到思考历史
                if search_result["status"] == "success":
                    result_content = search_result["content"]
                    collected_info.append(result_content)
                    self.thinking_engine.add_search_result(search_query, result_content)
                else:
                    error_info = f"搜索失败: {search_result.get('content', '未知错误')}"
                    self.thinking_engine.add_search_result(search_query, error_info)
        
        # 生成最终答案
        final_answer = self._generate_final_answer(query, collected_info)
        
        return final_answer
    
    def _generate_final_answer(self, query: str, collected_info: List[str]) -> str:
        """
        基于收集的信息生成最终答案
        
        参数:
            query: 原始查询
            collected_info: 收集的信息列表
            
        返回:
            str: 最终答案
        """
        if not collected_info:
            return "抱歉，没有找到相关信息来回答您的问题。"
        
        try:
            # 构建最终答案生成提示
            context = "\n\n".join(collected_info)
            thinking_summary = self.thinking_engine.get_thinking_summary()
            
            final_prompt = f"""
            基于以下研究过程和收集的信息，请生成一个全面、准确的答案：
            
            原始问题: {query}
            
            思考过程摘要:
            {thinking_summary}
            
            收集的信息:
            {context}
            
            请提供一个结构化、详细的答案，包括：
            1. 直接回答用户的问题
            2. 提供支持性的详细信息
            3. 如果适用，包含相关的背景信息
            
            答案应该准确、完整且易于理解。
            """
            
            # 调用LLM生成最终答案
            response = self.llm.invoke([HumanMessage(content=final_prompt)])
            final_answer = response.content if hasattr(response, 'content') else str(response)
            
            # 验证答案质量
            validation_result = self.answer_validator.validate_answer(query, final_answer, context)
            
            if not validation_result["is_valid"]:
                print(f"[{self.tool_name}] 答案质量验证失败，分数: {validation_result['score']:.2f}")
                # 可以选择重新生成或添加改进建议
                suggestions = self.answer_validator.suggest_improvements(query, final_answer, validation_result)
                if suggestions:
                    final_answer += f"\n\n注意: {'; '.join(suggestions)}"
            
            return final_answer
            
        except Exception as e:
            print(f"[{self.tool_name}] 生成最终答案失败: {e}")
            # 返回简单的信息汇总
            return f"基于研究收集的信息：\n\n" + "\n\n".join(collected_info[:3])
    
    def get_research_stats(self) -> Dict[str, Any]:
        """
        获取研究统计信息
        
        返回:
            Dict[str, Any]: 统计信息
        """
        return {
            "search_count": self.search_count,
            "max_search_limit": self.max_search_limit,
            "reasoning_steps": len(self.thinking_engine.get_reasoning_history()),
            "executed_queries": self.thinking_engine.get_executed_queries(),
            "performance_metrics": self.performance_metrics
        }
    
    def reset(self):
        """重置工具状态"""
        self.search_count = 0
        self.thinking_engine.reset()
        self.dual_path_searcher.reset_search_count()
        self._keywords_cache.clear()
        self.performance_metrics.clear()
    
    def close(self):
        """关闭资源"""
        super().close()
        
        # 关闭子工具
        if hasattr(self, 'hybrid_tool'):
            self.hybrid_tool.close()
        if hasattr(self, 'global_tool'):
            self.global_tool.close()
        if hasattr(self, 'local_tool'):
            self.local_tool.close()

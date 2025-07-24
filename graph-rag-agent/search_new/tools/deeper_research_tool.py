from typing import Dict, Any, List, AsyncGenerator
import json
import time
import traceback
import asyncio
import re
import os

from langchain_core.tools import BaseTool

from model.get_models import get_llm_model, get_embeddings_model
from config.neo4jdb import get_db_manager
from config.prompt import (
    system_template_build_graph,
    human_template_build_graph
)
from config.settings import (
    entity_types,
    relationship_types
)
from config.reasoning_prompts import RELEVANT_EXTRACTION_PROMPT
from search_new.reasoning.prompts.prompt_manager import kb_prompt
from graph.extraction.entity_extractor import EntityRelationExtractor
from search_new.tools.deep_research_tool import DeepResearchTool
from search_new.tools.hybrid_tool import HybridSearchTool
from search_new.reasoning.enhancers.community_enhancer import CommunityAwareSearchEnhancer
from search_new.reasoning.engines.thinking_engine import ThinkingEngine
from search_new.reasoning.enhancers.kg_builder import DynamicKnowledgeGraphBuilder
from search_new.reasoning.enhancers.evidence_tracker import EvidenceChainTracker
from search_new.reasoning.enhancers.exploration_chain import ChainOfExplorationSearcher
from search_new.reasoning.engines.validator import complexity_estimate


class DeeperResearchTool:
    """
    增强版深度研究工具
    
    整合社区感知、动态知识图谱和Chain of Exploration等功能，
    提供更全面的深度研究能力，并充分利用所有高级推理功能
    """
    
    def __init__(self, config=None, llm=None, embeddings=None, graph=None):
        """
        初始化增强版深度研究工具
        
        Args:
            config: 配置参数
            llm: 语言模型
            embeddings: 嵌入模型
            graph: 图数据库连接
        """
        # 关键词缓存
        self._keywords_cache = {}

        # 初始化基础组件
        self.llm = llm or get_llm_model()
        self.embeddings = embeddings or get_embeddings_model()
        self.graph = graph or get_db_manager()

        self.hybrid_tool = HybridSearchTool()
        
        # 初始化增强模块
        # 1. 社区感知搜索增强器
        self.community_search = CommunityAwareSearchEnhancer(self.llm)
        
        # 2. 动态知识图谱构建器
        self.knowledge_builder = DynamicKnowledgeGraphBuilder(self.llm)
        
        # 3. Chain of Exploration检索器
        self.chain_explorer = ChainOfExplorationSearcher(self.llm)
        
        # 4. 证据链跟踪器
        self.evidence_tracker = EvidenceChainTracker(self.llm)
        
        # 5. 继承原有的深度研究工具功能
        self.deep_research = DeepResearchTool()

        # 6. 查询生成器
        self.query_generator = self.deep_research.query_generator
        
        # 缓存设置
        self.enable_cache = True
        self.cache_dir = "./cache/deeper_research"
        
        # 确保缓存目录存在
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
                
        # 添加执行日志容器
        self.execution_logs = []
        
        # 添加性能指标跟踪
        self.performance_metrics = {"total_time": 0}
        
        # 记录当前查询的上下文信息
        self.current_query_context = {}
        
        # 记录已探索的查询分支
        self.explored_branches = {}
        
        # 共享思考引擎实例，避免重复初始化
        if hasattr(self.deep_research, 'thinking_engine'):
            self.thinking_engine = self.deep_research.thinking_engine
        else:
            self.thinking_engine = ThinkingEngine(self.llm)
            self.deep_research.thinking_engine = self.thinking_engine
        
        # 添加各种缓存字典
        self._search_cache = {}
        self._thinking_cache = {}
        self._contradiction_cache = {}
        self._hypotheses_cache = {}
        self._counter_cache = {}
        self._coe_cache = {}
        self._specific_coe_cache = {}
        self._contradiction_detailed_cache = {}
        self._stream_search_cache = {}
        self._stream_thinking_cache = {}
        self._subquery_cache = {}
    
    def _log(self, message):
        """记录执行日志"""
        self.execution_logs.append(message)
        # print(message)  # 可选：同时打印到控制台
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """从查询中提取关键词"""
        # 检查缓存
        if query in self._keywords_cache:
            return self._keywords_cache[query]

        keywords = self.hybrid_tool.extract_keywords(query)
        
        # 缓存结果
        self._keywords_cache[query] = keywords
        return keywords

    def _enhance_search_with_coe(self, query: str, keywords: Dict[str, List[str]]):
        """
        使用Chain of Exploration增强搜索

        Args:
            query: 用户查询
            keywords: 关键词字典

        Returns:
            Dict: 增强搜索结果
        """
        # 添加缓存检查
        cache_key = f"coe_search:{query}"
        if hasattr(self, '_coe_cache') and cache_key in self._coe_cache:
            return self._coe_cache[cache_key]

        # 获取社区感知上下文
        community_context = self.community_search.enhance_search_with_community_context(query, keywords)
        search_strategy = community_context.get("search_strategy", {})

        # 提取关注实体
        focus_entities = search_strategy.get("focus_entities", [])
        if not focus_entities:
            # 如果没有关注实体，从关键词提取
            focus_entities = keywords.get("high_level", []) + keywords.get("low_level", [])

        # 使用Chain of Exploration探索
        if focus_entities:
            # 添加缓存检查
            coe_cache_key = f"coe:{query}:{','.join(focus_entities[:3])}"
            if hasattr(self, '_specific_coe_cache') and coe_cache_key in self._specific_coe_cache:
                exploration_results = self._specific_coe_cache[coe_cache_key]
            else:
                exploration_results = self.chain_explorer.explore(
                    focus_entities[:3],  # 使用前3个关注实体作为起点
                    query
                )
                if not hasattr(self, '_specific_coe_cache'):
                    self._specific_coe_cache = {}
                self._specific_coe_cache[coe_cache_key] = exploration_results

            # 将探索结果添加到社区上下文
            community_context["exploration_results"] = exploration_results

            # 更新搜索策略
            discovered_entities = []
            for step in exploration_results.get("exploration_path", []):
                if step.get("step", 0) > 0:  # 跳过起始实体
                    discovered_entities.append(step.get("node_id", ""))

            if discovered_entities:
                search_strategy["discovered_entities"] = discovered_entities
                community_context["search_strategy"] = search_strategy

        # 缓存结果
        if not hasattr(self, '_coe_cache'):
            self._coe_cache = {}
        self._coe_cache[cache_key] = community_context

        return community_context

    def _create_multiple_reasoning_branches(self, query_id, hypotheses=None):
        """
        根据多个假设创建多个推理分支

        Args:
            query_id: 查询ID
            hypotheses: 假设列表

        Returns:
            Dict: 包含分支结果的字典
        """
        branch_results = {}

        # 避免重复生成假设
        if hypotheses is None:
            # 从query_id获取原始查询
            original_query = None
            for query, current_id in self.current_query_context.items():
                if current_id == query_id:
                    original_query = query
                    break

            if original_query is None:
                # 如果找不到原始查询，尝试从思考引擎获取
                if hasattr(self.thinking_engine, 'query'):
                    original_query = self.thinking_engine.query
                else:
                    # 兜底方案，使用一个空假设列表
                    self._log(f"\n[分支推理] 无法找到原始查询，无法生成假设")
                    return {}

            # 检查假设缓存
            if not hasattr(self, '_hypotheses_cache'):
                self._hypotheses_cache = {}

            if query_id in self._hypotheses_cache:
                hypotheses = self._hypotheses_cache[query_id]
            else:
                # 生成假设并缓存
                hypotheses = self.query_generator.generate_multiple_hypotheses(original_query, self.llm)
                self._hypotheses_cache[query_id] = hypotheses

        # 为每个假设创建一个推理分支
        for i, hypothesis in enumerate(hypotheses[:3]):  # 限制最多3个分支
            branch_name = f"branch_{i+1}"

            # 在思考引擎中创建推理分支
            if hasattr(self.thinking_engine, 'branch_reasoning'):
                self.thinking_engine.branch_reasoning(branch_name)

            # 记录分支创建
            self._log(f"\n[分支推理] 创建分支 {branch_name}: {hypothesis}")

            # 添加推理步骤
            step_id = self.evidence_tracker.add_reasoning_step(
                query_id,
                f"branch_{branch_name}",
                f"基于假设: {hypothesis} 创建推理分支"
            )

            # 记录分支信息
            self.explored_branches[branch_name] = {
                "hypothesis": hypothesis,
                "step_id": step_id,
                "evidence": []
            }

            # 在思考引擎中添加假设作为推理步骤
            self.thinking_engine.add_reasoning_step(
                f"探索假设: {hypothesis}"
            )

            # 应用反事实分析 - 仅对第一个分支进行
            if i == 0:
                # 缓存反事实分析
                counter_cache_key = f"counter:{query_id}:{hypothesis}"
                if hasattr(self, '_counter_cache') and counter_cache_key in self._counter_cache:
                    counter_analysis = self._counter_cache[counter_cache_key]
                else:
                    if hasattr(self.thinking_engine, 'counter_factual_analysis'):
                        counter_analysis = self.thinking_engine.counter_factual_analysis(
                            f"假设 {hypothesis} 不成立"
                        )
                    else:
                        counter_analysis = f"反事实分析: 如果假设 {hypothesis} 不成立，需要考虑其他可能性"

                    if not hasattr(self, '_counter_cache'):
                        self._counter_cache = {}
                    self._counter_cache[counter_cache_key] = counter_analysis

                # 记录反事实分析结果
                if hasattr(self.evidence_tracker, 'add_evidence'):
                    self.evidence_tracker.add_evidence(
                        f"counter_analysis_{i}",
                        counter_analysis
                    )

            branch_results[branch_name] = {
                "hypothesis": hypothesis,
                "step_id": step_id,
                "counter_analysis": counter_analysis if i == 0 else None
            }

        return branch_results

    def _detect_and_resolve_contradictions(self, query_id):
        """
        检测并处理信息矛盾

        Args:
            query_id: 查询ID

        Returns:
            Dict: 矛盾分析结果
        """
        # 添加缓存检查
        cache_key = f"contradiction:{query_id}"
        if hasattr(self, '_contradiction_detailed_cache') and cache_key in self._contradiction_detailed_cache:
            return self._contradiction_detailed_cache[cache_key]

        # 获取所有已收集的证据
        all_evidence = []
        if hasattr(self.evidence_tracker, 'get_reasoning_chain'):
            reasoning_chain = self.evidence_tracker.get_reasoning_chain(query_id)

            for step in reasoning_chain.get("steps", []):
                step_id = step.get("step_id", "")
                evidence_ids = step.get("evidence_ids", [])
                if evidence_ids:
                    all_evidence.extend(evidence_ids)

        # 检测矛盾
        contradictions = []
        if hasattr(self.evidence_tracker, 'detect_contradictions'):
            contradictions = self.evidence_tracker.detect_contradictions(all_evidence)

        if contradictions:
            self._log(f"\n[矛盾检测] 发现 {len(contradictions)} 个矛盾")

            # 记录矛盾分析
            contradiction_step_id = self.evidence_tracker.add_reasoning_step(
                query_id,
                "contradiction_analysis",
                f"分析 {len(contradictions)} 个信息矛盾"
            )

            # 解析每个矛盾
            for i, contradiction in enumerate(contradictions):
                contradiction_type = contradiction.get("type", "unknown")
                analysis = ""

                if contradiction_type == "numerical":
                    analysis = (f"数值矛盾: 在 '{contradiction.get('context', '')}' 中, "
                            f"发现值 {contradiction.get('value1')} 和 {contradiction.get('value2')}")
                elif contradiction_type == "semantic":
                    analysis = f"语义矛盾: {contradiction.get('analysis', '')}"

                # 记录矛盾证据
                if hasattr(self.evidence_tracker, 'add_evidence'):
                    self.evidence_tracker.add_evidence(
                        f"contradiction_{i}",
                        analysis
                    )

                self._log(f"\n[矛盾分析] {analysis}")

            result = {"contradictions": contradictions, "step_id": contradiction_step_id}
        else:
            result = {"contradictions": [], "step_id": None}

        # 缓存结果
        if not hasattr(self, '_contradiction_detailed_cache'):
            self._contradiction_detailed_cache = {}
        self._contradiction_detailed_cache[cache_key] = result

        return result

    def _generate_citations(self, answer, query_id):
        """
        为答案生成引用标记

        Args:
            answer: 原始答案
            query_id: 查询ID

        Returns:
            str: 带引用的答案
        """
        # 使用证据链跟踪器生成引用
        citation_result = self.evidence_tracker.generate_citations(answer)
        cited_answer = citation_result.get("cited_answer", answer)

        # 记录引用信息
        self._log(f"\n[引用生成] 添加了 {len(citation_result.get('citations', []))} 个引用")

        return cited_answer

    def search(self, query_input, **kwargs):
        """
        执行增强版深度研究

        Args:
            query_input: 查询输入，可以是字符串或字典
            **kwargs: 其他参数

        Returns:
            str: 研究结果
        """
        start_time = time.time()

        # 解析输入
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
            keywords = query_input.get("keywords", [])
        else:
            query = str(query_input)
            keywords = []

        # 生成查询ID
        query_id = f"deeper_research_{int(time.time())}"
        self.current_query_context[query] = query_id

        # 清空执行日志
        self.execution_logs = []

        try:
            self._log(f"\n[增强版深度研究] 开始研究: {query}")

            # 阶段1: 提取关键词
            if not keywords:
                keywords = self.extract_keywords(query)

            # 阶段2: 使用Chain of Exploration增强搜索
            enhanced_context = self._enhance_search_with_coe(query, keywords)

            # 阶段3: 创建多个推理分支
            branch_results = self._create_multiple_reasoning_branches(query_id)

            # 阶段4: 检测和解决矛盾
            contradiction_analysis = self._detect_and_resolve_contradictions(query_id)

            # 阶段5: 执行深度研究
            research_result = self.deep_research.search(query)

            # 阶段6: 生成引用
            cited_result = self._generate_citations(research_result, query_id)

            # 记录性能指标
            self.performance_metrics["total_time"] = time.time() - start_time
            self.performance_metrics["branches_created"] = len(branch_results)
            self.performance_metrics["contradictions_found"] = len(contradiction_analysis.get("contradictions", []))

            self._log(f"\n[增强版深度研究] 研究完成，用时 {self.performance_metrics['total_time']:.2f}秒")

            return cited_result

        except Exception as e:
            self._log(f"\n[增强版深度研究] 研究失败: {e}")
            error_msg = f"增强版深度研究过程中出现问题: {str(e)}"

            # 记录性能指标
            self.performance_metrics["total_time"] = time.time() - start_time
            self.performance_metrics["error"] = str(e)

            return error_msg

    def get_research_summary(self, query_id=None):
        """
        获取研究摘要

        Args:
            query_id: 查询ID，如果为None则使用最新的查询

        Returns:
            Dict: 研究摘要
        """
        if query_id is None:
            # 获取最新的查询ID
            if self.current_query_context:
                query_id = list(self.current_query_context.values())[-1]
            else:
                return {"error": "没有找到查询记录"}

        summary = {
            "query_id": query_id,
            "execution_logs": self.execution_logs,
            "performance_metrics": self.performance_metrics,
            "explored_branches": self.explored_branches,
            "timestamp": time.time()
        }

        return summary

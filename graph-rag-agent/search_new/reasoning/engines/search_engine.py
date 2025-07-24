from typing import List, Dict, Any, Optional
import time

from search_new.tools.local_tool import LocalSearchTool
from search_new.tools.global_tool import GlobalSearchTool
from search_new.tools.hybrid_tool import HybridSearchTool
from search_new.config.search_config import search_config


class QueryGenerator:
    """查询生成器，负责生成和优化搜索查询"""
    
    def __init__(self, llm):
        """
        初始化查询生成器
        
        参数:
            llm: 大语言模型实例
        """
        self.llm = llm
    
    def generate_follow_up_queries(self, original_query: str, context: str, max_queries: int = 3) -> List[str]:
        """
        基于原始查询和上下文生成后续查询
        
        参数:
            original_query: 原始查询
            context: 上下文信息
            max_queries: 最大查询数量
            
        返回:
            List[str]: 生成的后续查询列表
        """
        try:
            prompt = f"""
            基于原始查询和已有信息，生成{max_queries}个相关的后续查询来获取更多信息：
            
            原始查询: {original_query}
            
            已有信息:
            {context}
            
            请生成{max_queries}个具体的后续查询，每个查询应该：
            1. 与原始查询相关但角度不同
            2. 能够获取补充信息
            3. 具体明确，便于搜索
            
            请以列表形式返回查询，每行一个。
            """
            
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 提取查询
            queries = []
            for line in content.split('\n'):
                line = line.strip()
                # 移除序号和标点
                line = line.lstrip('0123456789.-) ').strip()
                if len(line) > 5 and line not in queries:
                    queries.append(line)
                    if len(queries) >= max_queries:
                        break
            
            return queries
            
        except Exception as e:
            print(f"[QueryGenerator] 生成后续查询失败: {e}")
            return []
    
    def refine_query(self, query: str, feedback: str) -> str:
        """
        基于反馈优化查询
        
        参数:
            query: 原始查询
            feedback: 反馈信息
            
        返回:
            str: 优化后的查询
        """
        try:
            prompt = f"""
            请基于反馈信息优化以下查询：
            
            原始查询: {query}
            反馈: {feedback}
            
            请提供一个更精确、更有效的查询。
            """
            
            response = self.llm.invoke(prompt)
            refined_query = response.content if hasattr(response, 'content') else str(response)
            
            return refined_query.strip()
            
        except Exception as e:
            print(f"[QueryGenerator] 查询优化失败: {e}")
            return query
    
    def expand_query(self, query: str) -> List[str]:
        """
        扩展查询，生成相关的查询变体
        
        参数:
            query: 原始查询
            
        返回:
            List[str]: 查询变体列表
        """
        try:
            prompt = f"""
            请为以下查询生成3个相关的查询变体，每个变体应该从不同角度探索相同的主题：
            
            原始查询: {query}
            
            请生成3个查询变体，每行一个。
            """
            
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 提取查询变体
            variants = []
            for line in content.split('\n'):
                line = line.strip()
                line = line.lstrip('0123456789.-) ').strip()
                if len(line) > 5 and line != query and line not in variants:
                    variants.append(line)
                    if len(variants) >= 3:
                        break
            
            return variants
            
        except Exception as e:
            print(f"[QueryGenerator] 查询扩展失败: {e}")
            return []


class DualPathSearcher:
    """双路径搜索器，结合本地和全局搜索策略"""
    
    def __init__(self, llm=None, embeddings=None):
        """
        初始化双路径搜索器
        
        参数:
            llm: 大语言模型实例
            embeddings: 嵌入模型实例
        """
        # 初始化搜索工具
        self.local_tool = LocalSearchTool()
        self.global_tool = GlobalSearchTool()
        self.hybrid_tool = HybridSearchTool()
        
        # 初始化查询生成器
        self.query_generator = QueryGenerator(llm or self.local_tool.llm)
        
        # 从配置加载参数
        reasoning_config = search_config.get_reasoning_config()
        self.max_search_limit = reasoning_config.get("max_search_limit", 10)
        
        # 搜索计数器
        self.search_count = 0
    
    def search(self, query: str, search_type: str = "auto") -> Dict[str, Any]:
        """
        执行搜索
        
        参数:
            query: 搜索查询
            search_type: 搜索类型 ("local", "global", "hybrid", "auto")
            
        返回:
            Dict[str, Any]: 搜索结果
        """
        if self.search_count >= self.max_search_limit:
            return {
                "status": "limit_reached",
                "content": "已达到最大搜索次数限制",
                "chunks": [],
                "doc_aggs": []
            }
        
        self.search_count += 1
        start_time = time.time()
        
        try:
            # 根据搜索类型选择工具
            if search_type == "auto":
                search_type = self._determine_search_type(query)
            
            if search_type == "local":
                result = self._local_search(query)
            elif search_type == "global":
                result = self._global_search(query)
            elif search_type == "hybrid":
                result = self._hybrid_search(query)
            else:
                # 默认使用混合搜索
                result = self._hybrid_search(query)
            
            # 添加搜索元信息
            result.update({
                "search_type": search_type,
                "search_time": time.time() - start_time,
                "search_count": self.search_count
            })
            
            return result
            
        except Exception as e:
            print(f"[DualPathSearcher] 搜索失败: {e}")
            return {
                "status": "error",
                "content": f"搜索失败: {str(e)}",
                "chunks": [],
                "doc_aggs": [],
                "search_type": search_type,
                "search_time": time.time() - start_time,
                "search_count": self.search_count
            }
    
    def _determine_search_type(self, query: str) -> str:
        """
        自动确定搜索类型
        
        参数:
            query: 搜索查询
            
        返回:
            str: 推荐的搜索类型
        """
        # 简单的启发式规则
        query_lower = query.lower()
        
        # 包含具体实体名称的查询倾向于本地搜索
        if any(keyword in query_lower for keyword in ["具体", "详细", "如何", "什么是", "定义"]):
            return "local"
        
        # 包含总结性词汇的查询倾向于全局搜索
        if any(keyword in query_lower for keyword in ["总结", "概述", "整体", "全面", "所有", "哪些"]):
            return "global"
        
        # 默认使用混合搜索
        return "hybrid"
    
    def _local_search(self, query: str) -> Dict[str, Any]:
        """执行本地搜索"""
        try:
            result = self.local_tool.search(query)
            
            return {
                "status": "success",
                "content": result,
                "chunks": [{"content_with_weight": result, "text": result}],
                "doc_aggs": []
            }
            
        except Exception as e:
            print(f"[DualPathSearcher] 本地搜索失败: {e}")
            return {
                "status": "error",
                "content": f"本地搜索失败: {str(e)}",
                "chunks": [],
                "doc_aggs": []
            }
    
    def _global_search(self, query: str) -> Dict[str, Any]:
        """执行全局搜索"""
        try:
            results = self.global_tool.search(query)
            
            # 将中间结果转换为chunks格式
            chunks = []
            for i, result in enumerate(results):
                chunks.append({
                    "content_with_weight": result,
                    "text": result,
                    "doc_id": f"global_result_{i}"
                })
            
            return {
                "status": "success",
                "content": "\n\n".join(results) if results else "未找到相关信息",
                "chunks": chunks,
                "doc_aggs": []
            }
            
        except Exception as e:
            print(f"[DualPathSearcher] 全局搜索失败: {e}")
            return {
                "status": "error",
                "content": f"全局搜索失败: {str(e)}",
                "chunks": [],
                "doc_aggs": []
            }
    
    def _hybrid_search(self, query: str) -> Dict[str, Any]:
        """执行混合搜索"""
        try:
            result = self.hybrid_tool.search(query)
            
            return {
                "status": "success",
                "content": result,
                "chunks": [{"content_with_weight": result, "text": result}],
                "doc_aggs": []
            }
            
        except Exception as e:
            print(f"[DualPathSearcher] 混合搜索失败: {e}")
            return {
                "status": "error",
                "content": f"混合搜索失败: {str(e)}",
                "chunks": [],
                "doc_aggs": []
            }
    
    def multi_query_search(self, queries: List[str], search_type: str = "auto") -> List[Dict[str, Any]]:
        """
        执行多查询搜索
        
        参数:
            queries: 查询列表
            search_type: 搜索类型
            
        返回:
            List[Dict[str, Any]]: 搜索结果列表
        """
        results = []
        
        for query in queries:
            if self.search_count >= self.max_search_limit:
                break
                
            result = self.search(query, search_type)
            results.append(result)
        
        return results
    
    def reset_search_count(self):
        """重置搜索计数器"""
        self.search_count = 0
    
    def get_search_stats(self) -> Dict[str, Any]:
        """获取搜索统计信息"""
        return {
            "search_count": self.search_count,
            "max_search_limit": self.max_search_limit,
            "remaining_searches": max(0, self.max_search_limit - self.search_count)
        }

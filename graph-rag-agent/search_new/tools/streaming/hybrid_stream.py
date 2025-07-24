from typing import AsyncGenerator, Dict, Any
import asyncio
import time

from search_new.tools.streaming.base_stream import BaseStreamSearchTool
from search_new.tools.streaming.local_stream import LocalSearchStreamTool
from search_new.tools.streaming.global_stream import GlobalSearchStreamTool
from search_new.tools.hybrid_tool import HybridSearchTool
from search_new.config.search_config import search_config


class HybridSearchStreamTool(BaseStreamSearchTool):
    """
    混合搜索流式工具
    
    结合本地搜索和全局搜索，提供流式输出功能，
    实时展示两种搜索策略的执行过程和结果融合
    """
    
    def __init__(self):
        """初始化混合搜索流式工具"""
        super().__init__(cache_dir=search_config.get_cache_dir("hybrid_stream"))
        
        # 初始化混合搜索工具
        self.hybrid_tool = HybridSearchTool()
        
        # 初始化子搜索工具（用于流式展示）
        self.local_stream = LocalSearchStreamTool()
        self.global_stream = GlobalSearchStreamTool()
        
        # 工具名称
        self.tool_name = "HybridSearchStreamTool"
    
    async def _execute_stream_search(self, query: str, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行混合搜索的流式输出
        
        参数:
            query: 搜索查询
            **kwargs: 其他参数
            
        生成:
            Dict[str, Any]: 搜索结果片段
        """
        # 阶段1: 开始搜索
        yield await self._stream_status_update("initializing", "初始化混合搜索...")
        await asyncio.sleep(0.1)
        
        # 阶段2: 查询分析
        yield await self._stream_status_update("analyzing_query", "分析查询类型...")
        
        # 模拟查询分析
        query_analysis = {
            "complexity": "medium",
            "entities_count": 2,
            "requires_local": True,
            "requires_global": True
        }
        
        yield {
            "type": "query_analysis",
            "content": query_analysis,
            "timestamp": time.time()
        }
        await asyncio.sleep(0.3)
        
        # 阶段3: 并行执行本地和全局搜索
        yield await self._stream_status_update("parallel_search", "并行执行本地和全局搜索...")
        yield await self._stream_progress_update(0.2)
        
        # 创建并行任务
        local_task = asyncio.create_task(self._execute_local_search_stream(query))
        global_task = asyncio.create_task(self._execute_global_search_stream(query))
        
        # 收集两个搜索的结果
        local_results = []
        global_results = []
        
        # 并行处理两个搜索流
        local_done = False
        global_done = False
        
        while not (local_done and global_done):
            # 检查本地搜索
            if not local_done:
                try:
                    local_result = await asyncio.wait_for(local_task.__anext__(), timeout=0.1)
                    local_results.append(local_result)
                    
                    yield {
                        "type": "local_search_update",
                        "content": local_result,
                        "timestamp": time.time()
                    }
                except (asyncio.TimeoutError, StopAsyncIteration):
                    local_done = True
            
            # 检查全局搜索
            if not global_done:
                try:
                    global_result = await asyncio.wait_for(global_task.__anext__(), timeout=0.1)
                    global_results.append(global_result)
                    
                    yield {
                        "type": "global_search_update", 
                        "content": global_result,
                        "timestamp": time.time()
                    }
                except (asyncio.TimeoutError, StopAsyncIteration):
                    global_done = True
            
            # 更新进度
            progress = 0.2 + (0.6 * ((len(local_results) + len(global_results)) / 20))
            yield await self._stream_progress_update(min(0.8, progress))
            
            await asyncio.sleep(0.1)
        
        # 阶段4: 结果融合
        yield await self._stream_status_update("merging_results", "融合搜索结果...")
        yield await self._stream_progress_update(0.8)
        
        # 模拟结果融合过程
        fusion_steps = [
            "分析本地搜索结果质量",
            "分析全局搜索结果质量", 
            "计算结果相似度",
            "应用融合策略",
            "生成最终答案"
        ]
        
        for i, step in enumerate(fusion_steps):
            yield {
                "type": "fusion_step",
                "content": step,
                "step": i + 1,
                "total_steps": len(fusion_steps),
                "timestamp": time.time()
            }
            await asyncio.sleep(0.2)
        
        # 阶段5: 执行实际混合搜索
        yield await self._stream_status_update("generating_final", "生成最终结果...")
        
        try:
            # 在后台执行实际搜索
            final_result = await self.search_async(query)
            
            # 阶段6: 流式输出最终结果
            yield await self._stream_status_update("outputting", "输出最终答案...")
            yield await self._stream_progress_update(0.9)
            
            # 流式输出最终结果
            async for chunk in self._stream_text_output(final_result, "final_answer"):
                yield chunk
            
            yield await self._stream_progress_update(1.0)
            
        except Exception as e:
            yield {
                "type": "error",
                "content": f"混合搜索过程中出现错误: {str(e)}",
                "timestamp": time.time()
            }
    
    async def _execute_local_search_stream(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """执行本地搜索流"""
        async for result in self.local_stream._execute_stream_search(query):
            yield result
    
    async def _execute_global_search_stream(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """执行全局搜索流"""
        async for result in self.global_stream._execute_stream_search(query):
            yield result
    
    def search(self, query: str, **kwargs) -> str:
        """
        同步混合搜索方法
        
        参数:
            query: 搜索查询
            **kwargs: 其他参数
            
        返回:
            str: 搜索结果
        """
        return self.hybrid_tool.search(query)
    
    def extract_keywords(self, query: str) -> Dict[str, Any]:
        """
        提取关键词
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, Any]: 关键词信息
        """
        return self.hybrid_tool.extract_keywords(query)
    
    async def search_with_strategy_comparison(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        带策略对比的搜索，展示不同搜索策略的效果
        
        参数:
            query: 搜索查询
            
        生成:
            Dict[str, Any]: 策略对比和结果
        """
        strategies = ["local_only", "global_only", "hybrid"]
        
        for strategy in strategies:
            yield {
                "type": "strategy_start",
                "strategy": strategy,
                "timestamp": time.time()
            }
            
            # 模拟不同策略的执行
            if strategy == "local_only":
                result = await self.local_stream.search_async(query)
            elif strategy == "global_only":
                result = await self.global_stream.search_async(query)
            else:
                result = await self.search_async(query)
            
            yield {
                "type": "strategy_result",
                "strategy": strategy,
                "result": result[:200] + "..." if len(result) > 200 else result,
                "full_length": len(result),
                "timestamp": time.time()
            }
            
            await asyncio.sleep(0.5)
        
        # 策略对比总结
        yield {
            "type": "strategy_comparison",
            "content": "策略对比完成，混合搜索通常能提供最全面的结果",
            "timestamp": time.time()
        }
    
    def get_search_stats(self) -> Dict[str, Any]:
        """
        获取搜索统计信息
        
        返回:
            Dict[str, Any]: 统计信息
        """
        base_stats = self.get_stream_status()
        hybrid_stats = self.hybrid_tool.get_performance_metrics()
        local_stats = self.local_stream.get_search_stats()
        global_stats = self.global_stream.get_search_stats()
        
        return {
            **base_stats,
            "hybrid_search_stats": hybrid_stats,
            "local_stream_stats": local_stats,
            "global_stream_stats": global_stats,
            "tool_name": self.tool_name
        }
    
    def close(self):
        """关闭资源"""
        super().close()
        if hasattr(self.hybrid_tool, 'close'):
            self.hybrid_tool.close()
        if hasattr(self.local_stream, 'close'):
            self.local_stream.close()
        if hasattr(self.global_stream, 'close'):
            self.global_stream.close()

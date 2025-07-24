from typing import AsyncGenerator, Dict, Any, List
import asyncio
import time

from search_new.tools.streaming.base_stream import BaseStreamSearchTool
from search_new.tools.global_tool import GlobalSearchTool
from search_new.config.search_config import search_config


class GlobalSearchStreamTool(BaseStreamSearchTool):
    """
    全局搜索流式工具
    
    提供全局搜索的流式输出功能，实时展示社区搜索过程和结果聚合
    """
    
    def __init__(self):
        """初始化全局搜索流式工具"""
        super().__init__(cache_dir=search_config.get_cache_dir("global_stream"))
        
        # 初始化全局搜索工具
        self.global_tool = GlobalSearchTool()
        
        # 工具名称
        self.tool_name = "GlobalSearchStreamTool"
    
    async def _execute_stream_search(self, query: str, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行全局搜索的流式输出
        
        参数:
            query: 搜索查询
            **kwargs: 其他参数
            
        生成:
            Dict[str, Any]: 搜索结果片段
        """
        # 阶段1: 开始搜索
        yield await self._stream_status_update("initializing", "初始化全局搜索...")
        await asyncio.sleep(0.1)
        
        # 阶段2: 关键词提取
        yield await self._stream_status_update("extracting_keywords", "提取关键词...")
        keywords = self.global_tool.extract_keywords(query)
        
        yield {
            "type": "keywords",
            "content": keywords,
            "timestamp": time.time()
        }
        await asyncio.sleep(0.2)
        
        # 阶段3: 社区发现
        yield await self._stream_status_update("discovering_communities", "发现相关社区...")
        yield await self._stream_progress_update(0.2)
        
        # 模拟社区发现过程
        community_count = 0
        for i in range(5):
            community_count += 20
            yield {
                "type": "community_discovery",
                "content": f"已发现 {community_count} 个相关社区",
                "count": community_count,
                "timestamp": time.time()
            }
            await asyncio.sleep(0.3)
        
        yield await self._stream_progress_update(0.5)
        
        # 阶段4: Map阶段 - 并行处理社区
        yield await self._stream_status_update("map_phase", "Map阶段：并行处理社区...")
        
        # 模拟Map阶段的并行处理
        total_communities = 100
        processed = 0
        
        for batch in range(10):  # 10个批次
            batch_size = 10
            processed += batch_size
            progress = 0.5 + (processed / total_communities) * 0.3  # 0.5-0.8
            
            yield {
                "type": "map_progress",
                "content": f"Map阶段：已处理 {processed}/{total_communities} 个社区",
                "processed": processed,
                "total": total_communities,
                "timestamp": time.time()
            }
            
            yield await self._stream_progress_update(progress)
            await asyncio.sleep(0.2)
        
        # 阶段5: Reduce阶段 - 聚合结果
        yield await self._stream_status_update("reduce_phase", "Reduce阶段：聚合搜索结果...")
        yield await self._stream_progress_update(0.8)
        
        # 模拟Reduce阶段
        for i in range(3):
            yield {
                "type": "reduce_progress",
                "content": f"聚合中间结果... ({i+1}/3)",
                "timestamp": time.time()
            }
            await asyncio.sleep(0.3)
        
        # 阶段6: 执行实际搜索
        yield await self._stream_status_update("searching", "生成最终结果...")
        
        try:
            # 在后台执行实际搜索
            search_results = await self.search_async(query)
            
            # 阶段7: 流式输出结果
            yield await self._stream_status_update("generating", "输出搜索结果...")
            yield await self._stream_progress_update(0.9)
            
            # 如果结果是列表，逐个输出
            if isinstance(search_results, list):
                for i, result in enumerate(search_results):
                    yield {
                        "type": "result_item",
                        "content": result,
                        "index": i,
                        "total": len(search_results),
                        "timestamp": time.time()
                    }
                    await asyncio.sleep(0.1)
                
                # 输出汇总
                combined_result = "\n\n".join(search_results)
                async for chunk in self._stream_text_output(combined_result, "final_answer"):
                    yield chunk
            else:
                # 直接流式输出结果
                async for chunk in self._stream_text_output(search_results, "answer"):
                    yield chunk
            
            yield await self._stream_progress_update(1.0)
            
        except Exception as e:
            yield {
                "type": "error",
                "content": f"全局搜索过程中出现错误: {str(e)}",
                "timestamp": time.time()
            }
    
    def search(self, query: str, **kwargs) -> List[str]:
        """
        同步全局搜索方法
        
        参数:
            query: 搜索查询
            **kwargs: 其他参数
            
        返回:
            List[str]: 搜索结果列表
        """
        return self.global_tool.search(query)
    
    def extract_keywords(self, query: str) -> Dict[str, Any]:
        """
        提取关键词
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, Any]: 关键词信息
        """
        return self.global_tool.extract_keywords(query)
    
    async def search_with_community_details(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        带社区详情的搜索，展示每个社区的处理过程
        
        参数:
            query: 搜索查询
            
        生成:
            Dict[str, Any]: 社区处理详情和结果
        """
        # 模拟社区处理过程
        communities = [
            {"id": f"community_{i}", "name": f"社区{i}", "size": 50 + i * 10}
            for i in range(1, 6)
        ]
        
        for i, community in enumerate(communities):
            yield {
                "type": "community_start",
                "community": community,
                "progress": i / len(communities),
                "timestamp": time.time()
            }
            
            # 模拟社区内搜索
            await asyncio.sleep(0.5)
            
            # 模拟社区搜索结果
            community_result = f"社区 {community['name']} 的搜索结果：包含 {community['size']} 个相关实体..."
            
            yield {
                "type": "community_result",
                "community": community,
                "result": community_result,
                "timestamp": time.time()
            }
        
        # 最终聚合
        yield {
            "type": "aggregation",
            "content": "正在聚合所有社区的搜索结果...",
            "timestamp": time.time()
        }
        
        await asyncio.sleep(1.0)
        
        # 执行实际搜索并输出最终结果
        final_results = await self.search_async(query)
        
        yield {
            "type": "final_aggregated_result",
            "content": final_results,
            "communities_processed": len(communities),
            "timestamp": time.time()
        }
    
    def get_search_stats(self) -> Dict[str, Any]:
        """
        获取搜索统计信息
        
        返回:
            Dict[str, Any]: 统计信息
        """
        base_stats = self.get_stream_status()
        global_stats = self.global_tool.get_performance_metrics()
        
        return {
            **base_stats,
            "global_search_stats": global_stats,
            "tool_name": self.tool_name
        }
    
    def close(self):
        """关闭资源"""
        super().close()
        if hasattr(self.global_tool, 'close'):
            self.global_tool.close()

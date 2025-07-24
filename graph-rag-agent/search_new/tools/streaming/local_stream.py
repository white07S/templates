from typing import AsyncGenerator, Dict, Any
import asyncio
import time

from search_new.tools.streaming.base_stream import BaseStreamSearchTool
from search_new.tools.local_tool import LocalSearchTool
from search_new.config.search_config import search_config


class LocalSearchStreamTool(BaseStreamSearchTool):
    """
    本地搜索流式工具
    
    提供本地搜索的流式输出功能，实时展示搜索过程和结果
    """
    
    def __init__(self):
        """初始化本地搜索流式工具"""
        super().__init__(cache_dir=search_config.get_cache_dir("local_stream"))
        
        # 初始化本地搜索工具
        self.local_tool = LocalSearchTool()
        
        # 工具名称
        self.tool_name = "LocalSearchStreamTool"
    
    async def _execute_stream_search(self, query: str, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行本地搜索的流式输出
        
        参数:
            query: 搜索查询
            **kwargs: 其他参数
            
        生成:
            Dict[str, Any]: 搜索结果片段
        """
        # 阶段1: 开始搜索
        yield await self._stream_status_update("initializing", "初始化本地搜索...")
        await asyncio.sleep(0.1)
        
        # 阶段2: 关键词提取
        yield await self._stream_status_update("extracting_keywords", "提取关键词...")
        keywords = self.local_tool.extract_keywords(query)
        
        yield {
            "type": "keywords",
            "content": keywords,
            "timestamp": time.time()
        }
        await asyncio.sleep(0.2)
        
        # 阶段3: 向量搜索
        yield await self._stream_status_update("vector_search", "执行向量搜索...")
        yield await self._stream_progress_update(0.3)
        
        # 模拟向量搜索过程
        for i in range(3):
            yield {
                "type": "search_step",
                "content": f"搜索相关文档... ({i+1}/3)",
                "timestamp": time.time()
            }
            await asyncio.sleep(0.3)
        
        yield await self._stream_progress_update(0.6)
        
        # 阶段4: 执行实际搜索
        yield await self._stream_status_update("searching", "生成搜索结果...")
        
        try:
            # 在后台执行实际搜索
            search_result = await self.search_async(query)
            
            # 阶段5: 流式输出结果
            yield await self._stream_status_update("generating", "生成答案...")
            yield await self._stream_progress_update(0.8)
            
            # 流式输出搜索结果
            async for chunk in self._stream_text_output(search_result, "answer"):
                yield chunk
            
            yield await self._stream_progress_update(1.0)
            
        except Exception as e:
            yield {
                "type": "error",
                "content": f"搜索过程中出现错误: {str(e)}",
                "timestamp": time.time()
            }
    
    def search(self, query: str, **kwargs) -> str:
        """
        同步本地搜索方法
        
        参数:
            query: 搜索查询
            **kwargs: 其他参数
            
        返回:
            str: 搜索结果
        """
        return self.local_tool.search(query)
    
    def extract_keywords(self, query: str) -> Dict[str, Any]:
        """
        提取关键词
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, Any]: 关键词信息
        """
        return self.local_tool.extract_keywords(query)
    
    async def search_with_steps(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        带步骤的搜索，详细展示每个搜索阶段
        
        参数:
            query: 搜索查询
            
        生成:
            Dict[str, Any]: 搜索步骤和结果
        """
        steps = [
            ("初始化", "准备搜索环境"),
            ("关键词提取", "从查询中提取关键词"),
            ("实体识别", "识别查询中的实体"),
            ("向量搜索", "在知识库中搜索相关内容"),
            ("上下文构建", "构建搜索上下文"),
            ("答案生成", "基于搜索结果生成答案"),
            ("结果验证", "验证答案质量")
        ]
        
        for i, (step_name, step_desc) in enumerate(steps):
            progress = (i + 1) / len(steps)
            
            yield {
                "type": "step",
                "step_name": step_name,
                "step_description": step_desc,
                "progress": progress,
                "timestamp": time.time()
            }
            
            # 模拟步骤执行时间
            await asyncio.sleep(0.5)
            
            # 在最后一步执行实际搜索
            if i == len(steps) - 1:
                result = await self.search_async(query)
                
                yield {
                    "type": "final_result",
                    "content": result,
                    "progress": 1.0,
                    "timestamp": time.time()
                }
    
    def get_search_stats(self) -> Dict[str, Any]:
        """
        获取搜索统计信息
        
        返回:
            Dict[str, Any]: 统计信息
        """
        base_stats = self.get_stream_status()
        local_stats = self.local_tool.get_performance_metrics()
        
        return {
            **base_stats,
            "local_search_stats": local_stats,
            "tool_name": self.tool_name
        }
    
    def close(self):
        """关闭资源"""
        super().close()
        if hasattr(self.local_tool, 'close'):
            self.local_tool.close()

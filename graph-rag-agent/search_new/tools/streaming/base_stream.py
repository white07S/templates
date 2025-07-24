from typing import AsyncGenerator, Dict, Any, Optional
import asyncio
import time
import json
from abc import ABC, abstractmethod

from search_new.tools.base import BaseSearchTool


class BaseStreamSearchTool(BaseSearchTool, ABC):
    """
    流式搜索工具基类
    
    提供流式搜索的基础功能，包括：
    - 异步搜索执行
    - 实时结果流式输出
    - 进度跟踪
    - 错误处理
    """
    
    def __init__(self, cache_dir: str = None):
        """
        初始化流式搜索工具
        
        参数:
            cache_dir: 缓存目录
        """
        super().__init__(cache_dir=cache_dir)
        
        # 流式搜索状态
        self.is_streaming = False
        self.stream_progress = 0.0
        self.stream_status = "ready"
        
        # 流式输出缓冲区
        self.stream_buffer = []
        self.buffer_size = 1000  # 缓冲区大小
        
        # 流式搜索配置
        self.stream_config = {
            "chunk_size": 100,      # 每次输出的字符数
            "delay": 0.1,           # 输出延迟（秒）
            "enable_progress": True, # 是否启用进度跟踪
            "enable_status": True   # 是否启用状态更新
        }
    
    async def search_stream(self, query: str, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行流式搜索
        
        参数:
            query: 搜索查询
            **kwargs: 其他参数
            
        生成:
            Dict[str, Any]: 流式搜索结果，包含类型、内容、进度等信息
        """
        self.is_streaming = True
        self.stream_progress = 0.0
        self.stream_status = "searching"
        
        try:
            # 发送开始信号
            yield {
                "type": "start",
                "query": query,
                "timestamp": time.time(),
                "status": "started"
            }
            
            # 执行具体的流式搜索逻辑
            async for result in self._execute_stream_search(query, **kwargs):
                yield result
            
            # 发送完成信号
            yield {
                "type": "complete",
                "status": "completed",
                "progress": 1.0,
                "timestamp": time.time()
            }
            
        except Exception as e:
            # 发送错误信号
            yield {
                "type": "error",
                "error": str(e),
                "status": "error",
                "timestamp": time.time()
            }
        finally:
            self.is_streaming = False
            self.stream_status = "ready"
    
    @abstractmethod
    async def _execute_stream_search(self, query: str, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行具体的流式搜索逻辑（子类实现）
        
        参数:
            query: 搜索查询
            **kwargs: 其他参数
            
        生成:
            Dict[str, Any]: 搜索结果片段
        """
        pass
    
    async def _stream_text_output(self, text: str, output_type: str = "content") -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式输出文本内容
        
        参数:
            text: 要输出的文本
            output_type: 输出类型
            
        生成:
            Dict[str, Any]: 文本片段
        """
        if not text:
            return
        
        chunk_size = self.stream_config["chunk_size"]
        delay = self.stream_config["delay"]
        
        # 按块输出文本
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            
            # 更新进度
            progress = min(1.0, (i + chunk_size) / len(text))
            
            yield {
                "type": output_type,
                "content": chunk,
                "progress": progress,
                "is_final": i + chunk_size >= len(text),
                "timestamp": time.time()
            }
            
            # 添加延迟以模拟实时输出
            if delay > 0:
                await asyncio.sleep(delay)
    
    async def _stream_progress_update(self, progress: float, status: str = None) -> Dict[str, Any]:
        """
        生成进度更新
        
        参数:
            progress: 进度值 (0.0-1.0)
            status: 状态描述
            
        返回:
            Dict[str, Any]: 进度更新信息
        """
        self.stream_progress = progress
        if status:
            self.stream_status = status
        
        return {
            "type": "progress",
            "progress": progress,
            "status": self.stream_status,
            "timestamp": time.time()
        }
    
    async def _stream_status_update(self, status: str, message: str = None) -> Dict[str, Any]:
        """
        生成状态更新
        
        参数:
            status: 状态
            message: 状态消息
            
        返回:
            Dict[str, Any]: 状态更新信息
        """
        self.stream_status = status
        
        return {
            "type": "status",
            "status": status,
            "message": message or status,
            "timestamp": time.time()
        }
    
    def get_stream_status(self) -> Dict[str, Any]:
        """
        获取当前流式搜索状态
        
        返回:
            Dict[str, Any]: 状态信息
        """
        return {
            "is_streaming": self.is_streaming,
            "progress": self.stream_progress,
            "status": self.stream_status,
            "buffer_size": len(self.stream_buffer)
        }
    
    def configure_stream(self, **config) -> None:
        """
        配置流式搜索参数
        
        参数:
            **config: 配置参数
        """
        self.stream_config.update(config)
    
    async def stop_stream(self) -> None:
        """停止流式搜索"""
        self.is_streaming = False
        self.stream_status = "stopped"
    
    def clear_stream_buffer(self) -> None:
        """清空流式输出缓冲区"""
        self.stream_buffer.clear()
    
    # 同步搜索方法的异步包装
    async def search_async(self, query: str, **kwargs) -> str:
        """
        异步执行搜索（非流式）
        
        参数:
            query: 搜索查询
            **kwargs: 其他参数
            
        返回:
            str: 搜索结果
        """
        # 在异步环境中执行同步搜索
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search, query)
    
    # 为了保持向后兼容性，保留原有的同步search方法
    def search(self, query: str, **kwargs) -> str:
        """
        同步搜索方法（由子类实现）
        
        参数:
            query: 搜索查询
            **kwargs: 其他参数
            
        返回:
            str: 搜索结果
        """
        # 这个方法由具体的子类实现
        raise NotImplementedError("子类必须实现search方法")
    
    def _setup_chains(self):
        """设置处理链（由子类实现）"""
        # 这个方法由具体的子类实现
        pass

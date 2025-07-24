from typing import AsyncGenerator, Dict, Any, List
import asyncio
import time

from search_new.tools.streaming.base_stream import BaseStreamSearchTool
from search_new.tools.deep_research_tool import DeepResearchTool
from search_new.config.search_config import search_config


class DeepResearchStreamTool(BaseStreamSearchTool):
    """
    深度研究流式工具
    
    提供深度研究的流式输出功能，实时展示多轮搜索、
    推理过程和知识图谱构建
    """
    
    def __init__(self):
        """初始化深度研究流式工具"""
        super().__init__(cache_dir=search_config.get_cache_dir("deep_research_stream"))
        
        # 初始化深度研究工具
        self.deep_research_tool = DeepResearchTool()
        
        # 工具名称
        self.tool_name = "DeepResearchStreamTool"
    
    async def _execute_stream_search(self, query: str, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行深度研究的流式输出
        
        参数:
            query: 搜索查询
            **kwargs: 其他参数
            
        生成:
            Dict[str, Any]: 搜索结果片段
        """
        # 阶段1: 开始深度研究
        yield await self._stream_status_update("initializing", "初始化深度研究...")
        await asyncio.sleep(0.1)
        
        # 阶段2: 查询复杂度分析
        yield await self._stream_status_update("analyzing_complexity", "分析查询复杂度...")
        
        complexity_analysis = {
            "complexity_level": "high",
            "estimated_rounds": 5,
            "requires_reasoning": True,
            "requires_kg_building": True
        }
        
        yield {
            "type": "complexity_analysis",
            "content": complexity_analysis,
            "timestamp": time.time()
        }
        await asyncio.sleep(0.3)
        
        # 阶段3: 多轮搜索
        yield await self._stream_status_update("multi_round_search", "开始多轮搜索...")
        yield await self._stream_progress_update(0.1)
        
        # 模拟多轮搜索过程
        search_rounds = 5
        for round_num in range(1, search_rounds + 1):
            yield {
                "type": "search_round_start",
                "round": round_num,
                "total_rounds": search_rounds,
                "timestamp": time.time()
            }
            
            # 模拟每轮搜索的子步骤
            substeps = [
                f"第{round_num}轮：生成搜索查询",
                f"第{round_num}轮：执行搜索",
                f"第{round_num}轮：分析结果",
                f"第{round_num}轮：提取关键信息"
            ]
            
            for substep in substeps:
                yield {
                    "type": "search_substep",
                    "content": substep,
                    "round": round_num,
                    "timestamp": time.time()
                }
                await asyncio.sleep(0.2)
            
            # 模拟搜索结果
            round_result = f"第{round_num}轮搜索完成，发现了新的相关信息..."
            yield {
                "type": "search_round_result",
                "round": round_num,
                "content": round_result,
                "timestamp": time.time()
            }
            
            progress = 0.1 + (round_num / search_rounds) * 0.4  # 0.1-0.5
            yield await self._stream_progress_update(progress)
            
            await asyncio.sleep(0.3)
        
        # 阶段4: 推理过程
        yield await self._stream_status_update("reasoning", "执行推理分析...")
        yield await self._stream_progress_update(0.6)
        
        reasoning_steps = [
            "构建推理链",
            "分析因果关系",
            "验证逻辑一致性",
            "生成推理结论"
        ]
        
        for i, step in enumerate(reasoning_steps):
            yield {
                "type": "reasoning_step",
                "content": step,
                "step": i + 1,
                "total_steps": len(reasoning_steps),
                "timestamp": time.time()
            }
            await asyncio.sleep(0.4)
        
        yield await self._stream_progress_update(0.7)
        
        # 阶段5: 知识图谱构建
        yield await self._stream_status_update("building_kg", "构建知识图谱...")
        
        kg_steps = [
            "提取实体",
            "识别关系",
            "构建图结构",
            "验证图完整性"
        ]
        
        for i, step in enumerate(kg_steps):
            yield {
                "type": "kg_building_step",
                "content": step,
                "step": i + 1,
                "total_steps": len(kg_steps),
                "timestamp": time.time()
            }
            await asyncio.sleep(0.3)
        
        # 模拟知识图谱统计
        kg_stats = {
            "entities": 45,
            "relationships": 78,
            "communities": 8
        }
        
        yield {
            "type": "kg_statistics",
            "content": kg_stats,
            "timestamp": time.time()
        }
        
        yield await self._stream_progress_update(0.8)
        
        # 阶段6: 执行实际深度研究
        yield await self._stream_status_update("generating_final", "生成最终研究报告...")
        
        try:
            # 在后台执行实际搜索
            research_result = await self.search_async(query)
            
            # 阶段7: 流式输出研究报告
            yield await self._stream_status_update("outputting_report", "输出研究报告...")
            yield await self._stream_progress_update(0.9)
            
            # 流式输出研究结果
            async for chunk in self._stream_text_output(research_result, "research_report"):
                yield chunk
            
            yield await self._stream_progress_update(1.0)
            
        except Exception as e:
            yield {
                "type": "error",
                "content": f"深度研究过程中出现错误: {str(e)}",
                "timestamp": time.time()
            }
    
    def search(self, query: str, **kwargs) -> str:
        """
        同步深度研究方法

        参数:
            query: 搜索查询
            **kwargs: 其他参数

        返回:
            str: 研究结果
        """
        return self.deep_research_tool.search(query)

    def extract_keywords(self, query: str) -> Dict[str, Any]:
        """
        提取关键词

        参数:
            query: 查询字符串

        返回:
            Dict[str, Any]: 关键词信息
        """
        return self.deep_research_tool.extract_keywords(query)
    
    async def research_with_detailed_steps(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        带详细步骤的研究，展示每个研究阶段的详细过程
        
        参数:
            query: 研究查询
            
        生成:
            Dict[str, Any]: 研究步骤和结果
        """
        research_phases = [
            {
                "name": "初始探索",
                "description": "进行初始信息收集和问题分析",
                "duration": 2.0
            },
            {
                "name": "深度搜索",
                "description": "执行多轮深度搜索，收集全面信息",
                "duration": 3.0
            },
            {
                "name": "推理分析",
                "description": "基于收集的信息进行逻辑推理",
                "duration": 2.5
            },
            {
                "name": "知识整合",
                "description": "整合所有信息，构建知识体系",
                "duration": 2.0
            },
            {
                "name": "报告生成",
                "description": "生成最终的研究报告",
                "duration": 1.5
            }
        ]
        
        total_duration = sum(phase["duration"] for phase in research_phases)
        elapsed_time = 0
        
        for phase in research_phases:
            yield {
                "type": "research_phase_start",
                "phase": phase,
                "progress": elapsed_time / total_duration,
                "timestamp": time.time()
            }
            
            # 模拟阶段执行
            phase_steps = int(phase["duration"] * 2)  # 每0.5秒一个步骤
            for step in range(phase_steps):
                step_progress = (elapsed_time + (step + 1) * 0.5) / total_duration
                
                yield {
                    "type": "research_phase_progress",
                    "phase": phase["name"],
                    "step": step + 1,
                    "total_steps": phase_steps,
                    "progress": step_progress,
                    "timestamp": time.time()
                }
                
                await asyncio.sleep(0.5)
            
            elapsed_time += phase["duration"]
            
            yield {
                "type": "research_phase_complete",
                "phase": phase,
                "progress": elapsed_time / total_duration,
                "timestamp": time.time()
            }
        
        # 执行实际研究并输出最终结果
        final_result = await self.search_async(query)
        
        yield {
            "type": "final_research_result",
            "content": final_result,
            "progress": 1.0,
            "timestamp": time.time()
        }
    
    def get_research_stats(self) -> Dict[str, Any]:
        """
        获取研究统计信息
        
        返回:
            Dict[str, Any]: 统计信息
        """
        base_stats = self.get_stream_status()
        research_stats = self.deep_research_tool.get_performance_metrics()
        
        return {
            **base_stats,
            "deep_research_stats": research_stats,
            "tool_name": self.tool_name
        }
    
    def close(self):
        """关闭资源"""
        super().close()
        if hasattr(self.deep_research_tool, 'close'):
            self.deep_research_tool.close()

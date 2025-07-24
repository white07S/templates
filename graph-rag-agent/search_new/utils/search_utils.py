import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Union
from config.settings import SEARCH_CACHE_CONFIG

class SearchUtils:
    """搜索相关的通用工具类"""
    
    @staticmethod
    def generate_cache_key(query: str, **kwargs) -> str:
        """
        生成缓存键
        
        参数:
            query: 查询字符串
            **kwargs: 其他参数
            
        返回:
            str: 缓存键
        """
        # 构建缓存键字符串
        key_parts = [query]
        
        # 添加关键词参数
        if 'keywords' in kwargs and kwargs['keywords']:
            key_parts.append(f"keywords:{','.join(sorted(kwargs['keywords']))}")
        
        if 'low_level_keywords' in kwargs and kwargs['low_level_keywords']:
            key_parts.append(f"low:{','.join(sorted(kwargs['low_level_keywords']))}")
            
        if 'high_level_keywords' in kwargs and kwargs['high_level_keywords']:
            key_parts.append(f"high:{','.join(sorted(kwargs['high_level_keywords']))}")
        
        # 添加其他参数
        for key, value in sorted(kwargs.items()):
            if key not in ['keywords', 'low_level_keywords', 'high_level_keywords'] and value is not None:
                key_parts.append(f"{key}:{str(value)}")
        
        # 生成MD5哈希
        cache_key = "||".join(key_parts)
        return hashlib.md5(cache_key.encode('utf-8')).hexdigest()
    
    @staticmethod
    def validate_search_result(result: Any) -> bool:
        """
        验证搜索结果是否有效
        
        参数:
            result: 搜索结果
            
        返回:
            bool: 是否有效
        """
        if not result:
            return False
            
        if isinstance(result, str):
            # 检查字符串长度和内容
            if len(result.strip()) < 10:
                return False
            
            # 检查是否包含错误信息
            error_patterns = [
                "抱歉，处理您的问题时遇到了错误",
                "技术原因:",
                "无法获取",
                "无法回答这个问题",
                "没有找到相关信息",
                "对不起，我不能"
            ]
            
            for pattern in error_patterns:
                if pattern in result:
                    return False
                    
            return True
            
        elif isinstance(result, (list, dict)):
            return len(result) > 0
            
        return True
    
    @staticmethod
    def format_search_context(chunks: List[str], 
                             entities: List[str], 
                             relationships: List[str],
                             communities: List[str] = None) -> str:
        """
        格式化搜索上下文信息
        
        参数:
            chunks: 文本块列表
            entities: 实体列表
            relationships: 关系列表
            communities: 社区信息列表
            
        返回:
            str: 格式化的上下文字符串
        """
        context_parts = []
        
        # 添加文本块信息
        if chunks:
            context_parts.append("### 相关文档片段")
            for i, chunk in enumerate(chunks, 1):
                context_parts.append(f"{i}. {chunk}")
            context_parts.append("")
        
        # 添加实体信息
        if entities:
            context_parts.append("### 相关实体")
            for i, entity in enumerate(entities, 1):
                context_parts.append(f"{i}. {entity}")
            context_parts.append("")
        
        # 添加关系信息
        if relationships:
            context_parts.append("### 相关关系")
            for i, rel in enumerate(relationships, 1):
                context_parts.append(f"{i}. {rel}")
            context_parts.append("")
        
        # 添加社区信息
        if communities:
            context_parts.append("### 社区总结")
            for i, community in enumerate(communities, 1):
                context_parts.append(f"{i}. {community}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    @staticmethod
    def extract_keywords_from_text(text: str, max_keywords: int = 10) -> List[str]:
        """
        从文本中提取关键词（简单实现）
        
        参数:
            text: 输入文本
            max_keywords: 最大关键词数量
            
        返回:
            List[str]: 关键词列表
        """
        # 简单的关键词提取（可以使用更复杂的NLP方法）
        import re
        
        # 移除标点符号并分词
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 过滤停用词（简单版本）
        stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '什么', '可以', '这个', '那个', '怎么', '为什么', '如何'
        }
        
        # 过滤停用词和短词
        keywords = [word for word in words if len(word) > 1 and word not in stop_words]
        
        # 统计词频
        word_count = {}
        for word in keywords:
            word_count[word] = word_count.get(word, 0) + 1
        
        # 按词频排序并返回前N个
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:max_keywords]]
    
    @staticmethod
    def merge_search_results(results: List[Dict[str, Any]], 
                           score_field: str = "score",
                           content_field: str = "content") -> List[Dict[str, Any]]:
        """
        合并多个搜索结果
        
        参数:
            results: 搜索结果列表
            score_field: 分数字段名
            content_field: 内容字段名
            
        返回:
            List[Dict[str, Any]]: 合并后的结果
        """
        merged = {}
        
        for result in results:
            if content_field in result:
                content = result[content_field]
                score = result.get(score_field, 0.0)
                
                if content in merged:
                    # 如果内容已存在，取最高分数
                    merged[content] = max(merged[content], score)
                else:
                    merged[content] = score
        
        # 转换回列表格式并排序
        merged_results = [
            {content_field: content, score_field: score}
            for content, score in merged.items()
        ]
        
        merged_results.sort(key=lambda x: x[score_field], reverse=True)
        return merged_results
    
    @staticmethod
    def calculate_search_metrics(start_time: float, 
                               query_time: float = 0,
                               llm_time: float = 0) -> Dict[str, float]:
        """
        计算搜索性能指标
        
        参数:
            start_time: 开始时间
            query_time: 数据库查询时间
            llm_time: LLM处理时间
            
        返回:
            Dict[str, float]: 性能指标
        """
        total_time = time.time() - start_time
        
        return {
            "total_time": total_time,
            "query_time": query_time,
            "llm_time": llm_time,
            "other_time": total_time - query_time - llm_time
        }
    
    @staticmethod
    def parse_query_input(query_input: Any) -> Dict[str, Any]:
        """
        解析查询输入，统一处理字符串和字典格式
        
        参数:
            query_input: 查询输入，可以是字符串或字典
            
        返回:
            Dict[str, Any]: 解析后的查询信息
        """
        if isinstance(query_input, dict):
            return {
                "query": query_input.get("query", ""),
                "keywords": query_input.get("keywords", []),
                "low_level_keywords": query_input.get("low_level_keywords", []),
                "high_level_keywords": query_input.get("high_level_keywords", []),
                "parameters": query_input.get("parameters", {})
            }
        else:
            return {
                "query": str(query_input),
                "keywords": [],
                "low_level_keywords": [],
                "high_level_keywords": [],
                "parameters": {}
            }
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
        """
        截断文本到指定长度
        
        参数:
            text: 输入文本
            max_length: 最大长度
            suffix: 截断后缀
            
        返回:
            str: 截断后的文本
        """
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def clean_search_query(query: str) -> str:
        """
        清理搜索查询字符串
        
        参数:
            query: 原始查询
            
        返回:
            str: 清理后的查询
        """
        # 移除多余的空白字符
        query = ' '.join(query.split())
        
        # 移除特殊字符（保留中文、英文、数字和基本标点）
        import re
        query = re.sub(r'[^\w\s\u4e00-\u9fff\?\!\.\,\;\:\-\(\)]', '', query)
        
        return query.strip()

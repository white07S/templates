import re
from typing import List, Optional


class TextProcessor:
    """文本处理工具类，提供各种文本提取和处理功能"""
    
    @staticmethod
    def extract_between(text: str, start_marker: str, end_marker: str) -> List[str]:
        """
        提取起始和结束标记之间的内容
        
        参数:
            text: 要搜索的文本
            start_marker: 起始标记
            end_marker: 结束标记
            
        返回:
            List[str]: 提取的内容字符串列表
        """
        pattern = re.escape(start_marker) + r"(.*?)" + re.escape(end_marker)
        return re.findall(pattern, text, flags=re.DOTALL)

    @staticmethod
    def extract_from_templates(text: str, templates: List[str], regex: bool = False) -> List[str]:
        """
        基于带占位符的模板提取内容
        
        参数:
            text: 要搜索的文本
            templates: 带{}占位符的模板字符串列表
            regex: 是否将模板作为正则表达式处理
            
        返回:
            List[str]: 提取的内容字符串列表
        """
        results = []
        
        for template in templates:
            if regex:
                # 直接使用模板作为正则表达式
                matches = re.findall(template, text, re.DOTALL)
                results.extend(matches)
            else:
                # 将模板转换为正则表达式（通过转义和替换占位符）
                pattern = template.replace("{}", "(.*?)")
                pattern = re.escape(pattern).replace("\\(\\*\\*\\?\\)", "(.*?)")
                matches = re.findall(pattern, text, re.DOTALL)
                results.extend(matches)
        
        return results

    @staticmethod
    def extract_sentences(text: str, max_sentences: Optional[int] = None) -> List[str]:
        """
        从文本中提取句子
        
        参数:
            text: 要提取句子的文本
            max_sentences: 最大提取句子数
            
        返回:
            List[str]: 句子列表
        """
        if not text:
            return []
        
        # 简单的句子分割（可以使用NLP库进行改进）
        sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_endings, text)
        
        # 移除空字符串
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if max_sentences:
            return sentences[:max_sentences]
        return sentences

    @staticmethod
    def clean_text(text: str) -> str:
        """
        清理文本，移除多余的空白字符
        
        参数:
            text: 输入文本
            
        返回:
            str: 清理后的文本
        """
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除首尾空白
        text = text.strip()
        
        return text

    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """
        从文本中提取关键词
        
        参数:
            text: 输入文本
            max_keywords: 最大关键词数量
            
        返回:
            List[str]: 关键词列表
        """
        if not text:
            return []
        
        # 简单的关键词提取（基于词频）
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 过滤停用词
        stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '什么', '可以', '这个', '那个', '怎么', '为什么', '如何',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        # 过滤停用词和短词
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # 统计词频
        word_count = {}
        for word in keywords:
            word_count[word] = word_count.get(word, 0) + 1
        
        # 按词频排序并返回前N个
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:max_keywords]]

    @staticmethod
    def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        将文本分割成重叠的块
        
        参数:
            text: 输入文本
            chunk_size: 每块的大小（字符数）
            overlap: 重叠字符数
            
        返回:
            List[str]: 文本块列表
        """
        if not text or chunk_size <= 0:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # 如果不是最后一块，尝试在句子边界处分割
            if end < len(text):
                # 寻找最后一个句号、问号或感叹号
                last_sentence_end = max(
                    chunk.rfind('.'),
                    chunk.rfind('?'),
                    chunk.rfind('!'),
                    chunk.rfind('。'),
                    chunk.rfind('？'),
                    chunk.rfind('！')
                )
                
                if last_sentence_end > chunk_size // 2:  # 确保块不会太小
                    chunk = chunk[:last_sentence_end + 1]
                    end = start + last_sentence_end + 1
            
            chunks.append(chunk.strip())
            
            # 计算下一个块的起始位置（考虑重叠）
            start = max(start + 1, end - overlap)
            
            # 如果剩余文本太短，直接添加并结束
            if len(text) - start < chunk_size // 2:
                if start < len(text):
                    chunks.append(text[start:].strip())
                break
        
        return [chunk for chunk in chunks if chunk]  # 移除空块

    @staticmethod
    def extract_entities_simple(text: str) -> List[str]:
        """
        简单的实体提取（基于大写字母开头的词组）
        
        参数:
            text: 输入文本
            
        返回:
            List[str]: 可能的实体列表
        """
        if not text:
            return []
        
        # 匹配大写字母开头的词组（可能是实体）
        entity_pattern = r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b'
        entities = re.findall(entity_pattern, text)
        
        # 过滤常见的非实体词
        common_words = {
            'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'And', 'Or', 'But',
            'In', 'On', 'At', 'To', 'For', 'Of', 'With', 'By', 'From', 'Up', 'About',
            'Into', 'Through', 'During', 'Before', 'After', 'Above', 'Below', 'Between'
        }
        
        # 过滤并去重
        filtered_entities = []
        seen = set()
        for entity in entities:
            if entity not in common_words and entity.lower() not in seen:
                filtered_entities.append(entity)
                seen.add(entity.lower())
        
        return filtered_entities

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        标准化文本中的空白字符
        
        参数:
            text: 输入文本
            
        返回:
            str: 标准化后的文本
        """
        if not text:
            return ""
        
        # 将所有空白字符（包括换行符、制表符等）替换为单个空格
        text = re.sub(r'\s+', ' ', text)
        
        # 移除首尾空白
        return text.strip()

    @staticmethod
    def remove_html_tags(text: str) -> str:
        """
        移除HTML标签
        
        参数:
            text: 包含HTML标签的文本
            
        返回:
            str: 移除HTML标签后的文本
        """
        if not text:
            return ""
        
        # 移除HTML标签
        clean_text = re.sub(r'<[^>]+>', '', text)
        
        # 标准化空白字符
        return TextProcessor.normalize_whitespace(clean_text)

    @staticmethod
    def extract_numbers(text: str) -> List[str]:
        """
        从文本中提取数字
        
        参数:
            text: 输入文本
            
        返回:
            List[str]: 数字列表
        """
        if not text:
            return []
        
        # 匹配整数和小数
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        return re.findall(number_pattern, text)

    @staticmethod
    def extract_dates(text: str) -> List[str]:
        """
        从文本中提取日期
        
        参数:
            text: 输入文本
            
        返回:
            List[str]: 日期列表
        """
        if not text:
            return []
        
        # 匹配常见的日期格式
        date_patterns = [
            r'\b\d{4}-\d{1,2}-\d{1,2}\b',  # YYYY-MM-DD
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{1,2}-\d{1,2}-\d{4}\b',  # MM-DD-YYYY
            r'\b\d{4}年\d{1,2}月\d{1,2}日\b',  # 中文日期格式
        ]
        
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text))
        
        return dates


# 为了保持向后兼容性，提供原有的函数接口
def extract_between(text: str, start_marker: str, end_marker: str) -> List[str]:
    """提取起始和结束标记之间的内容"""
    return TextProcessor.extract_between(text, start_marker, end_marker)


def extract_from_templates(text: str, templates: List[str], regex: bool = False) -> List[str]:
    """基于带占位符的模板提取内容"""
    return TextProcessor.extract_from_templates(text, templates, regex)


def extract_sentences(text: str, max_sentences: Optional[int] = None) -> List[str]:
    """从文本中提取句子"""
    return TextProcessor.extract_sentences(text, max_sentences)

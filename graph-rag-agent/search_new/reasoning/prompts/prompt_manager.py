from typing import List, Dict, Any
from collections import defaultdict
import logging


class PromptManager:
    """提示模板管理器，提供知识库信息格式化和token计算功能"""
    
    @staticmethod
    def num_tokens_from_string(text: str) -> int:
        """
        估算文本字符串中的token数量
        
        参数:
            text: 文本字符串
            
        返回:
            int: 估计的token数
        """
        try:
            from model.get_models import count_tokens
            return count_tokens(text)
        except:
            # 简单备用方法：中文字符按1个token计算，英文按4个字符1个token计算
            chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
            other_chars = len(text) - chinese_chars
            return chinese_chars + other_chars // 4

    @staticmethod
    def kb_prompt(kbinfos: Dict[str, List[Dict[str, Any]]], max_tokens: int = 4096) -> List[str]:
        """
        将知识库信息格式化为结构化提示
        
        参数:
            kbinfos: 包含chunks和文档聚合的字典
            max_tokens: 结果提示的最大token数
            
        返回:
            List[str]: 格式化的信息块列表
        """
        # 从chunks中提取content_with_weight
        knowledges = []
        for ck in kbinfos.get("chunks", []):
            content = ck.get("content_with_weight", ck.get("text", ""))
            if content:
                knowledges.append(content)
        
        # 限制总token数
        used_token_count = 0
        chunks_num = 0
        for i, c in enumerate(knowledges):
            used_token_count += PromptManager.num_tokens_from_string(c)
            chunks_num += 1
            if max_tokens * 0.97 < used_token_count:
                knowledges = knowledges[:i]
                logging.warning(f"未将所有检索结果放入提示: {i+1}/{len(knowledges)}")
                break
        
        # 获取文档信息
        doc_aggs = kbinfos.get("doc_aggs", [])
        docs = {d.get("doc_id", ""): d for d in doc_aggs}
        
        # 按文档组织chunks
        doc2chunks = defaultdict(lambda: {"chunks": [], "meta": {}})
        
        for ck in knowledges:
            # 尝试从chunk中提取文档信息
            doc_id = ""
            if isinstance(ck, dict):
                doc_id = ck.get("doc_id", "")
                content = ck.get("content_with_weight", ck.get("text", ""))
            else:
                content = str(ck)
            
            # 如果没有文档ID，使用默认文档
            if not doc_id:
                doc_id = "未知文档"
            
            doc2chunks[doc_id]["chunks"].append(content)
            
            # 添加文档元数据
            if doc_id in docs:
                doc_info = docs[doc_id]
                doc2chunks[doc_id]["meta"].update({
                    "标题": doc_info.get("title", ""),
                    "作者": doc_info.get("author", ""),
                    "类型": doc_info.get("doc_type", ""),
                    "摘要": doc_info.get("summary", "")
                })
        
        # 格式化最终知识块
        formatted_knowledges = []
        for doc_name, cks_meta in doc2chunks.items():
            txt = f"\nDocument: {doc_name} \n"
            
            # 添加元数据
            for k, v in cks_meta["meta"].items():
                if v:  # 只添加非空的元数据
                    txt += f"{k}: {v}\n"
                
            txt += "Relevant fragments as following:\n"
            
            # 添加chunk内容
            for chunk in cks_meta["chunks"]:
                txt += f"{chunk}\n"
                
            formatted_knowledges.append(txt)
        
        # 如果没有找到chunks
        if not formatted_knowledges:
            return ["在知识库中未找到相关信息。"]
            
        return formatted_knowledges

    @staticmethod
    def format_search_results(results: List[Dict[str, Any]], max_length: int = 2000) -> str:
        """
        格式化搜索结果为可读文本
        
        参数:
            results: 搜索结果列表
            max_length: 最大文本长度
            
        返回:
            str: 格式化的搜索结果
        """
        if not results:
            return "未找到相关搜索结果。"
        
        formatted_parts = []
        current_length = 0
        
        for i, result in enumerate(results, 1):
            # 提取结果内容
            content = ""
            if isinstance(result, dict):
                content = result.get("content", result.get("text", str(result)))
            else:
                content = str(result)
            
            # 格式化单个结果
            formatted_result = f"结果 {i}:\n{content}\n"
            
            # 检查长度限制
            if current_length + len(formatted_result) > max_length:
                if formatted_parts:  # 如果已经有内容，就停止添加
                    break
                else:  # 如果是第一个结果但太长，就截断
                    remaining_length = max_length - current_length - 20  # 留一些空间给省略号
                    if remaining_length > 100:
                        content = content[:remaining_length] + "..."
                        formatted_result = f"结果 {i}:\n{content}\n"
            
            formatted_parts.append(formatted_result)
            current_length += len(formatted_result)
        
        return "\n".join(formatted_parts)

    @staticmethod
    def format_reasoning_context(reasoning_steps: List[str], max_steps: int = 10) -> str:
        """
        格式化推理步骤为上下文
        
        参数:
            reasoning_steps: 推理步骤列表
            max_steps: 最大步骤数
            
        返回:
            str: 格式化的推理上下文
        """
        if not reasoning_steps:
            return "暂无推理历史。"
        
        # 限制步骤数量
        steps_to_show = reasoning_steps[-max_steps:] if len(reasoning_steps) > max_steps else reasoning_steps
        
        formatted_steps = []
        for i, step in enumerate(steps_to_show, 1):
            formatted_steps.append(f"步骤 {i}: {step}")
        
        return "\n".join(formatted_steps)

    @staticmethod
    def format_entity_context(entities: List[Dict[str, Any]], max_entities: int = 20) -> str:
        """
        格式化实体信息为上下文
        
        参数:
            entities: 实体信息列表
            max_entities: 最大实体数量
            
        返回:
            str: 格式化的实体上下文
        """
        if not entities:
            return "未找到相关实体。"
        
        # 限制实体数量
        entities_to_show = entities[:max_entities]
        
        formatted_entities = []
        for entity in entities_to_show:
            if isinstance(entity, dict):
                entity_id = entity.get("id", "未知实体")
                description = entity.get("description", "无描述")
                formatted_entities.append(f"- {entity_id}: {description}")
            else:
                formatted_entities.append(f"- {str(entity)}")
        
        return "\n".join(formatted_entities)

    @staticmethod
    def format_relationship_context(relationships: List[Dict[str, Any]], max_rels: int = 15) -> str:
        """
        格式化关系信息为上下文
        
        参数:
            relationships: 关系信息列表
            max_rels: 最大关系数量
            
        返回:
            str: 格式化的关系上下文
        """
        if not relationships:
            return "未找到相关关系。"
        
        # 限制关系数量
        rels_to_show = relationships[:max_rels]
        
        formatted_rels = []
        for rel in rels_to_show:
            if isinstance(rel, dict):
                description = rel.get("description", str(rel))
                weight = rel.get("weight", "")
                if weight:
                    formatted_rels.append(f"- {description} (权重: {weight})")
                else:
                    formatted_rels.append(f"- {description}")
            else:
                formatted_rels.append(f"- {str(rel)}")
        
        return "\n".join(formatted_rels)

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
    def merge_contexts(contexts: List[str], separator: str = "\n\n") -> str:
        """
        合并多个上下文
        
        参数:
            contexts: 上下文列表
            separator: 分隔符
            
        返回:
            str: 合并后的上下文
        """
        # 过滤空上下文
        valid_contexts = [ctx for ctx in contexts if ctx and ctx.strip()]
        
        if not valid_contexts:
            return ""
        
        return separator.join(valid_contexts)


# 为了保持向后兼容性，提供原有的函数接口
def num_tokens_from_string(text: str) -> int:
    """估算文本字符串中的token数量"""
    return PromptManager.num_tokens_from_string(text)


def kb_prompt(kbinfos: Dict[str, List[Dict[str, Any]]], max_tokens: int = 4096) -> List[str]:
    """将知识库信息格式化为结构化提示"""
    return PromptManager.kb_prompt(kbinfos, max_tokens)

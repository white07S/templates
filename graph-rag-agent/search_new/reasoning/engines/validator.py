import re
from typing import Dict, Any, List, Optional
from search_new.config.search_config import search_config


def complexity_estimate(query: str) -> str:
    """
    估算查询复杂度
    
    参数:
        query: 查询字符串
        
    返回:
        str: 复杂度等级 ("simple", "medium", "complex")
    """
    if not query:
        return "simple"
    
    # 复杂度指标
    complexity_indicators = {
        "simple": ["什么是", "定义", "介绍"],
        "medium": ["如何", "为什么", "比较", "分析"],
        "complex": ["评估", "综合", "深入分析", "全面", "系统性"]
    }
    
    query_lower = query.lower()
    
    # 检查复杂度指标
    for level, indicators in complexity_indicators.items():
        if any(indicator in query_lower for indicator in indicators):
            return level
    
    # 基于查询长度判断
    if len(query) < 10:
        return "simple"
    elif len(query) < 30:
        return "medium"
    else:
        return "complex"


class AnswerValidator:
    """答案验证器，负责验证答案的质量和完整性"""
    
    def __init__(self, llm):
        """
        初始化答案验证器
        
        参数:
            llm: 大语言模型实例
        """
        self.llm = llm
        
        # 从配置加载验证参数
        validation_config = search_config.get_validation_config()
        self.min_answer_length = validation_config.get("min_answer_length", 50)
        self.max_answer_length = validation_config.get("max_answer_length", 5000)
        self.relevance_threshold = validation_config.get("relevance_threshold", 0.7)
    
    def validate_answer(self, query: str, answer: str, context: str = "") -> Dict[str, Any]:
        """
        验证答案质量
        
        参数:
            query: 原始查询
            answer: 待验证的答案
            context: 上下文信息
            
        返回:
            Dict[str, Any]: 验证结果
        """
        validation_result = {
            "is_valid": True,
            "score": 0.0,
            "issues": [],
            "suggestions": []
        }
        
        # 1. 基本格式检查
        format_score = self._check_format(answer, validation_result)
        
        # 2. 内容相关性检查
        relevance_score = self._check_relevance(query, answer, validation_result)
        
        # 3. 完整性检查
        completeness_score = self._check_completeness(query, answer, validation_result)
        
        # 4. 事实一致性检查
        consistency_score = self._check_consistency(answer, context, validation_result)
        
        # 计算总分
        validation_result["score"] = (
            format_score * 0.2 + 
            relevance_score * 0.4 + 
            completeness_score * 0.3 + 
            consistency_score * 0.1
        )
        
        # 判断是否有效
        validation_result["is_valid"] = (
            validation_result["score"] >= self.relevance_threshold and
            len(validation_result["issues"]) == 0
        )
        
        return validation_result
    
    def _check_format(self, answer: str, result: Dict[str, Any]) -> float:
        """检查答案格式"""
        score = 1.0
        
        # 检查长度
        if len(answer) < self.min_answer_length:
            result["issues"].append(f"答案过短（{len(answer)}字符，最少需要{self.min_answer_length}字符）")
            result["suggestions"].append("请提供更详细的答案")
            score -= 0.3
        
        if len(answer) > self.max_answer_length:
            result["issues"].append(f"答案过长（{len(answer)}字符，最多{self.max_answer_length}字符）")
            result["suggestions"].append("请精简答案内容")
            score -= 0.2
        
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
            if pattern in answer:
                result["issues"].append("答案包含错误信息")
                result["suggestions"].append("请重新生成答案")
                score -= 0.5
                break
        
        # 检查结构化程度
        if "###" in answer or "####" in answer:
            score += 0.1  # 有结构化标题加分
        
        return max(0.0, score)
    
    def _check_relevance(self, query: str, answer: str, result: Dict[str, Any]) -> float:
        """检查答案相关性"""
        try:
            # 使用LLM评估相关性
            relevance_prompt = f"""
            请评估以下答案与查询的相关性，给出0-1之间的分数：
            
            查询: {query}
            答案: {answer[:500]}...
            
            请只返回一个0-1之间的数字，表示相关性分数。
            """
            
            response = self.llm.invoke(relevance_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 提取分数
            score_match = re.search(r'0\.\d+|1\.0|0|1', content)
            if score_match:
                score = float(score_match.group())
            else:
                score = 0.5  # 默认分数
            
            if score < self.relevance_threshold:
                result["issues"].append(f"答案相关性不足（{score:.2f}，需要≥{self.relevance_threshold}）")
                result["suggestions"].append("请提供更相关的答案")
            
            return score
            
        except Exception as e:
            print(f"[AnswerValidator] 相关性检查失败: {e}")
            return 0.5
    
    def _check_completeness(self, query: str, answer: str, result: Dict[str, Any]) -> float:
        """检查答案完整性"""
        score = 1.0
        
        # 基于查询复杂度检查完整性
        complexity = complexity_estimate(query)
        
        if complexity == "simple":
            # 简单查询需要基本信息
            if len(answer) < 100:
                result["suggestions"].append("可以提供更多基础信息")
                score -= 0.1
        elif complexity == "medium":
            # 中等查询需要详细解释
            if len(answer) < 200:
                result["suggestions"].append("可以提供更详细的解释")
                score -= 0.2
        elif complexity == "complex":
            # 复杂查询需要全面分析
            if len(answer) < 300:
                result["suggestions"].append("可以提供更全面的分析")
                score -= 0.3
        
        # 检查是否有引用数据部分
        if "#### 引用数据" not in answer and "引用" not in answer:
            result["suggestions"].append("建议添加数据来源引用")
            score -= 0.1
        
        return max(0.0, score)
    
    def _check_consistency(self, answer: str, context: str, result: Dict[str, Any]) -> float:
        """检查答案与上下文的一致性"""
        if not context:
            return 1.0  # 没有上下文时不检查一致性
        
        try:
            # 使用LLM检查一致性
            consistency_prompt = f"""
            请检查答案与上下文信息是否一致，给出0-1之间的分数：
            
            上下文: {context[:500]}...
            答案: {answer[:500]}...
            
            请只返回一个0-1之间的数字，表示一致性分数。
            """
            
            response = self.llm.invoke(consistency_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 提取分数
            score_match = re.search(r'0\.\d+|1\.0|0|1', content)
            if score_match:
                score = float(score_match.group())
            else:
                score = 0.8  # 默认分数
            
            if score < 0.7:
                result["issues"].append(f"答案与上下文不一致（{score:.2f}）")
                result["suggestions"].append("请确保答案与提供的信息一致")
            
            return score
            
        except Exception as e:
            print(f"[AnswerValidator] 一致性检查失败: {e}")
            return 0.8
    
    def suggest_improvements(self, query: str, answer: str, validation_result: Dict[str, Any]) -> List[str]:
        """
        基于验证结果提供改进建议
        
        参数:
            query: 原始查询
            answer: 答案
            validation_result: 验证结果
            
        返回:
            List[str]: 改进建议列表
        """
        suggestions = validation_result.get("suggestions", []).copy()
        
        # 基于分数提供额外建议
        score = validation_result.get("score", 0.0)
        
        if score < 0.5:
            suggestions.append("答案质量较低，建议重新生成")
        elif score < 0.7:
            suggestions.append("答案质量一般，可以进一步改进")
        
        # 基于查询类型提供建议
        complexity = complexity_estimate(query)
        if complexity == "complex" and len(answer) < 500:
            suggestions.append("复杂查询需要更详细的分析和解释")
        
        return list(set(suggestions))  # 去重
    
    def is_answer_sufficient(self, query: str, answer: str, min_score: float = None) -> bool:
        """
        判断答案是否足够好
        
        参数:
            query: 查询
            answer: 答案
            min_score: 最小分数阈值
            
        返回:
            bool: 是否足够好
        """
        if min_score is None:
            min_score = self.relevance_threshold
        
        validation_result = self.validate_answer(query, answer)
        return validation_result["score"] >= min_score and validation_result["is_valid"]
    
    def compare_answers(self, query: str, answers: List[str]) -> Dict[str, Any]:
        """
        比较多个答案的质量
        
        参数:
            query: 查询
            answers: 答案列表
            
        返回:
            Dict[str, Any]: 比较结果
        """
        if not answers:
            return {"best_answer": "", "best_score": 0.0, "rankings": []}
        
        # 验证所有答案
        validations = []
        for i, answer in enumerate(answers):
            validation = self.validate_answer(query, answer)
            validation["index"] = i
            validation["answer"] = answer
            validations.append(validation)
        
        # 按分数排序
        validations.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "best_answer": validations[0]["answer"],
            "best_score": validations[0]["score"],
            "rankings": validations
        }

import re
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from config.reasoning_prompts import REASON_PROMPT
from search_new.reasoning.nlp.text_processor import TextProcessor
from search_new.config.search_config import search_config


class ThinkingEngine:
    """
    思考引擎类：负责管理多轮迭代的思考过程
    提供思考历史管理和转换功能
    """
    
    def __init__(self, llm):
        """
        初始化思考引擎
        
        参数:
            llm: 大语言模型实例，用于生成思考内容
        """
        self.llm = llm
        self.all_reasoning_steps = []
        self.msg_history = []
        self.executed_search_queries = []
        self.hypotheses = []       # 存储假设
        self.verification_chain = [] # 验证步骤
        self.reasoning_tree = {}   # 推理树结构
        self.current_branch = "main" # 当前推理分支
        
        # 从配置加载参数
        reasoning_config = search_config.get_reasoning_config()
        self.max_iterations = reasoning_config.get("max_iterations", 5)
        self.thinking_depth = reasoning_config.get("thinking_depth", 3)
    
    def initialize_with_query(self, query: str):
        """使用初始查询初始化思考历史"""
        self.all_reasoning_steps = []
        self.msg_history = [{"role": "user", "content": f'问题:"{query}"\n'}]
        self.executed_search_queries = []
        self.hypotheses = []
        self.verification_chain = []
        self.reasoning_tree = {"main": []} # 初始化主分支
        self.current_branch = "main"
    
    def add_reasoning_step(self, step: str):
        """添加推理步骤到历史中"""
        if step and step.strip():
            self.all_reasoning_steps.append(step.strip())
            self.reasoning_tree[self.current_branch].append(step.strip())
            
            # 添加到消息历史
            self.msg_history.append({"role": "assistant", "content": step.strip()})
    
    def add_executed_query(self, query: str):
        """添加已执行的搜索查询"""
        if query and query.strip():
            self.executed_search_queries.append(query.strip())
    
    def get_reasoning_history(self) -> List[str]:
        """获取推理历史"""
        return self.all_reasoning_steps.copy()
    
    def get_executed_queries(self) -> List[str]:
        """获取已执行的查询历史"""
        return self.executed_search_queries.copy()
    
    def extract_queries(self, text: str) -> List[str]:
        """
        从AI响应中提取搜索查询
        
        参数:
            text: AI生成的文本
            
        返回:
            List[str]: 提取的查询列表
        """
        if not text:
            return []
        
        queries = []
        
        # 方法1: 提取引号中的内容
        quoted_queries = re.findall(r'"([^"]+)"', text)
        for query in quoted_queries:
            if len(query.strip()) > 5:  # 过滤太短的查询
                queries.append(query.strip())
        
        # 方法2: 提取搜索标记之间的内容
        search_patterns = [
            r'搜索[:：]\s*([^\n]+)',
            r'查询[:：]\s*([^\n]+)',
            r'检索[:：]\s*([^\n]+)',
            r'search[:：]\s*([^\n]+)',
        ]
        
        for pattern in search_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_query = match.strip().strip('"\'')
                if len(clean_query) > 5:
                    queries.append(clean_query)
        
        # 方法3: 提取问号结尾的句子
        question_queries = re.findall(r'([^。！？\n]+[？?])', text)
        for query in question_queries:
            clean_query = query.strip()
            if len(clean_query) > 10 and clean_query not in queries:
                queries.append(clean_query)
        
        # 去重并限制数量
        unique_queries = []
        seen = set()
        for query in queries:
            if query.lower() not in seen and len(unique_queries) < 3:
                unique_queries.append(query)
                seen.add(query.lower())
        
        return unique_queries
    
    def remove_query_tags(self, text: str) -> str:
        """移除查询标记，清理文本"""
        if not text:
            return ""
        
        # 移除常见的查询标记
        patterns_to_remove = [
            r'<search>.*?</search>',
            r'<query>.*?</query>',
            r'<think>.*?</think>',
        ]
        
        cleaned_text = text
        for pattern in patterns_to_remove:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL)
        
        return cleaned_text.strip()
    
    def generate_next_query(self) -> Dict[str, Any]:
        """
        生成下一步搜索查询
        
        返回:
            Dict: 包含查询和状态信息的字典
        """
        # 检查是否已达到最大迭代次数
        if len(self.executed_search_queries) >= self.max_iterations:
            return {
                "status": "max_iterations_reached",
                "content": "已达到最大搜索次数，准备生成最终答案。",
                "queries": []
            }
        
        # 构建消息历史
        formatted_messages = []
        
        # 添加系统提示
        formatted_messages.append(SystemMessage(content=REASON_PROMPT))
        
        # 添加历史消息
        for msg in self.msg_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                formatted_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                formatted_messages.append(AIMessage(content=content))
        
        # 添加当前推理状态
        if self.executed_search_queries:
            executed_info = f"\n已执行的搜索: {', '.join(self.executed_search_queries)}"
            formatted_messages.append(HumanMessage(content=executed_info))
        
        try:
            # 调用LLM生成查询
            msg = self.llm.invoke(formatted_messages)
            query_think = msg.content if hasattr(msg, 'content') else str(msg)
            
            # 清理响应
            query_think = re.sub(r"<think>.*</think>", "", query_think, flags=re.DOTALL)
            if not query_think:
                return {"status": "empty", "content": None, "queries": []}
                
            # 更新思考过程
            clean_think = self.remove_query_tags(query_think)
            self.add_reasoning_step(query_think)
            
            # 从AI响应中提取搜索查询
            queries = self.extract_queries(query_think)
            
            # 如果没有生成搜索查询，检查是否应该结束
            if not queries:
                # 检查是否包含最终答案标记
                if "**回答**" in query_think or "足够的信息" in query_think:
                    return {
                        "status": "answer_ready", 
                        "content": query_think,
                        "queries": []
                    }
                
                # 没有明确结束标志，就继续
                return {
                    "status": "no_query", 
                    "content": query_think,
                    "queries": []
                }
            
            # 返回结果状态和查询
            return {
                "status": "has_query", 
                "content": query_think,
                "queries": queries
            }
            
        except Exception as e:
            print(f"[ThinkingEngine] 生成查询失败: {e}")
            return {
                "status": "error",
                "content": f"生成查询时出错: {str(e)}",
                "queries": []
            }
    
    def add_search_result(self, query: str, result: str):
        """
        添加搜索结果到思考历史
        
        参数:
            query: 搜索查询
            result: 搜索结果
        """
        if query and result:
            search_info = f"搜索查询: {query}\n搜索结果: {result}"
            self.add_reasoning_step(search_info)
    
    def generate_hypothesis(self, context: str) -> str:
        """
        基于上下文生成假设
        
        参数:
            context: 上下文信息
            
        返回:
            str: 生成的假设
        """
        try:
            hypothesis_prompt = f"""
            基于以下信息，生成一个合理的假设来回答用户的问题：
            
            {context}
            
            请生成一个具体的假设，并说明支持这个假设的理由。
            """
            
            response = self.llm.invoke([HumanMessage(content=hypothesis_prompt)])
            hypothesis = response.content if hasattr(response, 'content') else str(response)
            
            # 添加到假设列表
            self.hypotheses.append(hypothesis)
            
            return hypothesis
            
        except Exception as e:
            print(f"[ThinkingEngine] 生成假设失败: {e}")
            return "无法生成假设"
    
    def verify_hypothesis(self, hypothesis: str) -> str:
        """
        验证假设
        
        参数:
            hypothesis: 要验证的假设
            
        返回:
            str: 验证结果
        """
        try:
            verification_prompt = f"""
            请验证以下假设的合理性：
            
            假设: {hypothesis}
            
            基于已有的推理历史: {' '.join(self.all_reasoning_steps[-3:])}
            
            请分析这个假设是否合理，并提供验证结果。
            """
            
            response = self.llm.invoke([HumanMessage(content=verification_prompt)])
            verification = response.content if hasattr(response, 'content') else str(response)
            
            # 添加到验证链
            self.verification_chain.append({
                "hypothesis": hypothesis,
                "verification": verification
            })
            
            return verification
            
        except Exception as e:
            print(f"[ThinkingEngine] 验证假设失败: {e}")
            return "无法验证假设"
    
    def get_thinking_summary(self) -> str:
        """获取思考过程的摘要"""
        if not self.all_reasoning_steps:
            return "暂无思考历史"
        
        # 获取最近的几个推理步骤
        recent_steps = self.all_reasoning_steps[-self.thinking_depth:]
        
        summary_parts = []
        summary_parts.append("=== 思考过程摘要 ===")
        
        for i, step in enumerate(recent_steps, 1):
            # 截断过长的步骤
            truncated_step = TextProcessor.truncate_text(step, max_length=200)
            summary_parts.append(f"{i}. {truncated_step}")
        
        if self.executed_search_queries:
            summary_parts.append(f"\n已执行搜索: {', '.join(self.executed_search_queries)}")
        
        return "\n".join(summary_parts)
    
    def reset(self):
        """重置思考引擎状态"""
        self.all_reasoning_steps = []
        self.msg_history = []
        self.executed_search_queries = []
        self.hypotheses = []
        self.verification_chain = []
        self.reasoning_tree = {"main": []}
        self.current_branch = "main"
    
    def export_reasoning_trace(self) -> Dict[str, Any]:
        """导出完整的推理轨迹"""
        return {
            "reasoning_steps": self.all_reasoning_steps,
            "executed_queries": self.executed_search_queries,
            "hypotheses": self.hypotheses,
            "verification_chain": self.verification_chain,
            "reasoning_tree": self.reasoning_tree,
            "current_branch": self.current_branch
        }

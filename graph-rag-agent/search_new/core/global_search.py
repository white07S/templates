from typing import List, Dict, Any
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.prompt import MAP_SYSTEM_PROMPT, REDUCE_SYSTEM_PROMPT
from config.neo4jdb import get_db_manager
from search_new.config.search_config import search_config
from search_new.utils.search_utils import SearchUtils


class GlobalSearch:
    """
    全局搜索类：使用Neo4j和LangChain实现基于Map-Reduce模式的全局搜索功能
    
    该类主要用于在整个知识图谱范围内进行搜索，采用以下步骤：
    1. 获取指定层级的所有社区数据
    2. Map阶段：为每个社区生成中间结果
    3. Reduce阶段：整合所有中间结果生成最终答案
    """
    
    def __init__(self, llm=None, response_type: str = None):
        """
        初始化全局搜索类
        
        参数:
            llm: 大语言模型实例，如果为None则从模型管理器获取
            response_type: 响应类型，如果为None则使用配置中的默认值
        """
        # 初始化模型
        if llm is None:
            from model.get_models import get_llm_model
            llm = get_llm_model()
            
        self.llm = llm
        
        # 设置响应类型
        self.response_type = response_type or search_config.get_response_type()
        
        # 使用数据库连接管理
        db_manager = get_db_manager()
        self.graph = db_manager.get_graph()
        
        # 从配置加载搜索参数
        self._load_search_config()
        
    def _load_search_config(self):
        """从配置加载搜索参数"""
        global_config = search_config.get_global_search_config()
        
        self.default_level = global_config.get("default_level", 0)
        self.batch_size = global_config.get("batch_size", 10)
        self.max_communities = global_config.get("max_communities", 100)
        
    def _get_community_data(self, level: int) -> List[dict]:
        """
        获取指定层级的社区数据
        
        参数:
            level: 社区层级
            
        返回:
            List[dict]: 社区数据字典列表
        """
        try:
            # 限制社区数量以避免性能问题
            result = self.graph.query(
                """
                MATCH (c:__Community__)
                WHERE c.level = $level
                RETURN {communityId:c.id, full_content:c.full_content} AS output
                ORDER BY c.weight DESC
                LIMIT $max_communities
                """,
                params={"level": level, "max_communities": self.max_communities},
            )
            
            print(f"[GlobalSearch] 获取到 {len(result)} 个层级 {level} 的社区")
            return result
            
        except Exception as e:
            print(f"[GlobalSearch] 获取社区数据失败: {e}")
            return []
    
    def _process_communities(self, query: str, communities: List[dict]) -> List[str]:
        """
        处理社区数据生成中间结果（Map阶段）
        
        参数:
            query: 搜索查询字符串
            communities: 社区数据列表
            
        返回:
            List[str]: 中间结果列表
        """
        if not communities:
            return []
            
        # 设置Map阶段的提示模板
        map_prompt = ChatPromptTemplate.from_messages([
            ("system", MAP_SYSTEM_PROMPT),
            ("human", """
                ---数据表格--- 
                {context_data}
                
                用户的问题是：
                {question}
                """),
        ])
        
        # 创建Map阶段的处理链
        map_chain = map_prompt | self.llm | StrOutputParser()
        
        # 批量处理社区
        results = []
        total_communities = len(communities)
        
        print(f"[GlobalSearch] 开始处理 {total_communities} 个社区...")
        
        # 分批处理以提高效率
        for i in range(0, total_communities, self.batch_size):
            batch = communities[i:i + self.batch_size]
            batch_results = []
            
            for community in tqdm(batch, desc=f"处理批次 {i//self.batch_size + 1}"):
                try:
                    # 获取社区内容
                    community_data = community.get("output", {})
                    
                    # 如果社区内容为空，跳过
                    if not community_data or not community_data.get("full_content"):
                        continue
                    
                    # 调用LLM处理社区数据
                    response = map_chain.invoke({
                        "question": query,
                        "context_data": community_data
                    })
                    
                    # 验证响应
                    if SearchUtils.validate_search_result(response):
                        batch_results.append(response)
                        print(f"[GlobalSearch] 社区 {community_data.get('communityId', 'unknown')} 处理完成")
                    else:
                        print(f"[GlobalSearch] 社区 {community_data.get('communityId', 'unknown')} 响应无效，跳过")
                        
                except Exception as e:
                    print(f"[GlobalSearch] 处理社区失败: {e}")
                    continue
            
            results.extend(batch_results)
            print(f"[GlobalSearch] 批次 {i//self.batch_size + 1} 完成，获得 {len(batch_results)} 个有效结果")
        
        print(f"[GlobalSearch] Map阶段完成，共获得 {len(results)} 个中间结果")
        return results
    
    def _reduce_results(self, query: str, intermediate_results: List[str]) -> str:
        """
        整合中间结果生成最终答案（Reduce阶段）
        
        参数:
            query: 搜索查询字符串
            intermediate_results: 中间结果列表
            
        返回:
            str: 最终答案
        """
        if not intermediate_results:
            return "抱歉，未能找到相关信息来回答您的问题。"
        
        # 如果只有一个结果，直接返回
        if len(intermediate_results) == 1:
            return intermediate_results[0]
        
        try:
            # 设置Reduce阶段的提示模板
            reduce_prompt = ChatPromptTemplate.from_messages([
                ("system", REDUCE_SYSTEM_PROMPT),
                ("human", """
                    ---中间分析结果--- 
                    {context_data}
                    
                    用户的问题是：
                    {question}
                    
                    请基于以上分析结果，生成一个综合性的回答。
                    """),
            ])
            
            # 创建Reduce阶段的处理链
            reduce_chain = reduce_prompt | self.llm | StrOutputParser()
            
            # 合并中间结果
            combined_context = "\n\n".join([
                f"### 分析结果 {i+1}\n{result}" 
                for i, result in enumerate(intermediate_results)
            ])
            
            # 截断过长的上下文
            combined_context = SearchUtils.truncate_text(combined_context, max_length=8000)
            
            # 调用LLM生成最终答案
            final_response = reduce_chain.invoke({
                "question": query,
                "context_data": combined_context,
                "response_type": self.response_type
            })
            
            # 验证最终响应
            if not SearchUtils.validate_search_result(final_response):
                # 如果最终响应无效，返回最好的中间结果
                return max(intermediate_results, key=len)
            
            return final_response
            
        except Exception as e:
            print(f"[GlobalSearch] Reduce阶段失败: {e}")
            # 返回最长的中间结果作为备选
            return max(intermediate_results, key=len) if intermediate_results else "搜索过程中出现问题。"
    
    def search(self, query: str, level: int = None) -> str:
        """
        执行全局搜索
        
        参数:
            query: 搜索查询字符串
            level: 要搜索的社区层级，如果为None则使用默认层级
            
        返回:
            str: 生成的最终答案
        """
        # 清理查询
        query = SearchUtils.clean_search_query(query)
        
        # 使用默认层级
        if level is None:
            level = self.default_level
        
        print(f"[GlobalSearch] 开始全局搜索，查询: '{query}', 层级: {level}")
        
        try:
            # 获取社区数据
            communities = self._get_community_data(level)
            
            if not communities:
                return "抱歉，未能找到相关的社区信息来回答您的问题。"
            
            # 处理社区数据（Map阶段）
            intermediate_results = self._process_communities(query, communities)
            
            if not intermediate_results:
                return "抱歉，未能从社区数据中提取到相关信息。"
            
            # 生成最终答案（Reduce阶段）
            final_answer = self._reduce_results(query, intermediate_results)
            
            print(f"[GlobalSearch] 全局搜索完成")
            return final_answer
            
        except Exception as e:
            print(f"[GlobalSearch] 全局搜索失败: {e}")
            return f"搜索过程中出现问题: {str(e)}"
    
    def search_with_map_only(self, query: str, level: int = None) -> List[str]:
        """
        只执行Map阶段，返回中间结果列表
        
        参数:
            query: 搜索查询字符串
            level: 要搜索的社区层级
            
        返回:
            List[str]: 中间结果列表
        """
        # 清理查询
        query = SearchUtils.clean_search_query(query)
        
        # 使用默认层级
        if level is None:
            level = self.default_level
        
        print(f"[GlobalSearch] 开始Map阶段搜索，查询: '{query}', 层级: {level}")
        
        try:
            # 获取社区数据
            communities = self._get_community_data(level)
            
            if not communities:
                return []
            
            # 处理社区数据（仅Map阶段）
            intermediate_results = self._process_communities(query, communities)
            
            print(f"[GlobalSearch] Map阶段完成，获得 {len(intermediate_results)} 个结果")
            return intermediate_results
            
        except Exception as e:
            print(f"[GlobalSearch] Map阶段搜索失败: {e}")
            return []
    
    def get_community_summary(self, level: int = None) -> Dict[str, Any]:
        """
        获取社区摘要信息
        
        参数:
            level: 社区层级
            
        返回:
            Dict[str, Any]: 社区摘要信息
        """
        if level is None:
            level = self.default_level
        
        try:
            result = self.graph.query(
                """
                MATCH (c:__Community__)
                WHERE c.level = $level
                RETURN count(c) as total_communities,
                       avg(c.weight) as avg_weight,
                       max(c.weight) as max_weight,
                       min(c.weight) as min_weight
                """,
                params={"level": level}
            )
            
            if result:
                return result[0]
            else:
                return {}
                
        except Exception as e:
            print(f"[GlobalSearch] 获取社区摘要失败: {e}")
            return {}
    
    def close(self):
        """关闭资源连接"""
        # 连接由数据库管理器管理，这里不需要手动关闭
        pass
            
    def __enter__(self):
        """上下文管理器入口"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

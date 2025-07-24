from typing import Dict, Any, List
import pandas as pd
from neo4j import Result
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.prompt import LC_SYSTEM_PROMPT
from config.neo4jdb import get_db_manager
from search_new.config.search_config import search_config
from search_new.utils.search_utils import SearchUtils


class LocalSearch:
    """
    本地搜索类：使用Neo4j和LangChain实现基于向量检索的本地搜索功能
    
    该类通过向量相似度搜索在知识图谱中查找相关内容，并生成回答
    主要功能包括：
    1. 基于向量相似度的文本检索
    2. 社区内容和关系的检索
    3. 使用LLM生成最终答案
    """
    
    def __init__(self, llm=None, embeddings=None, response_type: str = None):
        """
        初始化本地搜索类
        
        参数:
            llm: 大语言模型实例，如果为None则从模型管理器获取
            embeddings: 向量嵌入模型，如果为None则从模型管理器获取
            response_type: 响应类型，如果为None则使用配置中的默认值
        """
        # 初始化模型
        if llm is None:
            from model.get_models import get_llm_model
            llm = get_llm_model()
        if embeddings is None:
            from model.get_models import get_embeddings_model
            embeddings = get_embeddings_model()
            
        self.llm = llm
        self.embeddings = embeddings
        
        # 设置响应类型
        self.response_type = response_type or search_config.get_response_type()
        
        # 获取数据库连接管理器
        db_manager = get_db_manager()
        self.driver = db_manager.get_driver()
        self.neo4j_uri = db_manager.neo4j_uri
        self.neo4j_username = db_manager.neo4j_username
        self.neo4j_password = db_manager.neo4j_password
        
        # 从配置加载检索参数
        self._load_search_config()
        
        # 初始化社区节点权重
        self._init_community_weights()
    
    def _load_search_config(self):
        """从配置加载搜索参数"""
        local_config = search_config.get_local_search_config()
        
        self.top_chunks = local_config.get("top_chunks", 10)
        self.top_communities = local_config.get("top_communities", 2)
        self.top_outside_rels = local_config.get("top_outside_rels", 10)
        self.top_inside_rels = local_config.get("top_inside_rels", 10)
        self.top_entities = local_config.get("top_entities", 10)
        self.index_name = local_config.get("index_name", "vector")
    
    def _init_community_weights(self):
        """初始化社区节点权重"""
        try:
            # 获取数据库管理器并执行查询
            db_manager = get_db_manager()
            result = db_manager.execute_query("""
                MATCH (n:__Community__)
                SET n.community_rank = coalesce(n.weight, 0)
                RETURN count(n) as updated_count
            """)
            
            if not result.empty:
                updated_count = result.iloc[0]['updated_count']
                print(f"[LocalSearch] 已更新 {updated_count} 个社区节点的权重")

        except Exception as e:
            print(f"[LocalSearch] 初始化社区权重失败: {e}")

    @property
    def retrieval_query(self) -> str:
        """
        获取Neo4j检索查询语句

        返回:
            str: Cypher查询语句，用于检索相关内容
        """
        return """
        WITH collect(node) as nodes
        WITH
        collect {
            UNWIND nodes as n
            MATCH (n)<-[:MENTIONS]-(c:__Chunk__)
            WITH distinct c, count(distinct n) as freq
            RETURN {id:c.id, text: c.text} AS chunkText
            ORDER BY freq DESC
            LIMIT $topChunks
        } AS text_mapping,
        collect {
            UNWIND nodes as n
            MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
            WITH distinct c, c.community_rank as rank, c.weight AS weight
            RETURN c.summary
            ORDER BY rank, weight DESC
            LIMIT $topCommunities
        } AS report_mapping,
        collect {
            UNWIND nodes as n
            MATCH (n)-[r]-(m:__Entity__)
            WHERE NOT m IN nodes
            RETURN r.description AS descriptionText
            ORDER BY r.weight DESC
            LIMIT $topOutsideRels
        } as outsideRels,
        collect {
            UNWIND nodes as n
            MATCH (n)-[r]-(m:__Entity__)
            WHERE m IN nodes
            RETURN r.description AS descriptionText
            ORDER BY r.weight DESC
            LIMIT $topInsideRels
        } as insideRels,
        collect {
            UNWIND nodes as n
            RETURN n.description AS descriptionText
        } as entities
        RETURN {
            Chunks: text_mapping,
            Reports: report_mapping,
            Relationships: outsideRels + insideRels,
            Entities: entities
        } AS text, 1.0 AS score, {} AS metadata
        """
    
    def as_retriever(self, **kwargs):
        """
        返回检索器实例，用于链式调用
        
        参数:
            **kwargs: 额外的检索参数
            
        返回:
            检索器实例
        """
        # 生成包含所有检索参数的查询
        final_query = self.retrieval_query.replace("$topChunks", str(self.top_chunks))\
            .replace("$topCommunities", str(self.top_communities))\
            .replace("$topOutsideRels", str(self.top_outside_rels))\
            .replace("$topInsideRels", str(self.top_inside_rels))

        # 初始化向量存储
        vector_store = Neo4jVector.from_existing_index(
            self.embeddings,
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
            index_name=self.index_name,
            retrieval_query=final_query
        )
        
        # 返回检索器
        return vector_store.as_retriever(
            search_kwargs={"k": self.top_entities, **kwargs}
        )
    
    def search(self, query: str, **kwargs) -> str:
        """
        执行本地搜索
        
        参数:
            query: 搜索查询字符串
            **kwargs: 额外的搜索参数
            
        返回:
            str: 生成的最终答案
        """
        # 清理查询
        query = SearchUtils.clean_search_query(query)
        
        # 初始化对话提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            ("human", """
                ---分析报告--- 
                请注意，下面提供的分析报告按**重要性降序排列**。

                {context}

                用户的问题是：
                {input}

                请按以下格式输出回答：
                1. 使用三级标题(###)标记主题
                2. 主要内容用清晰的段落展示
                3. 最后必须用"#### 引用数据"标记引用部分，列出用到的数据来源
                """
             )
        ])
        
        # 创建搜索链
        chain = prompt | self.llm | StrOutputParser()
        
        # 构建检索查询参数
        search_params = {
            "topChunks": self.top_chunks,
            "topCommunities": self.top_communities,
            "topOutsideRels": self.top_outside_rels,
            "topInsideRels": self.top_inside_rels,
        }
        search_params.update(kwargs)
        
        try:
            # 生成包含所有检索参数的查询
            final_query = self.retrieval_query.replace("$topChunks", str(self.top_chunks))\
                .replace("$topCommunities", str(self.top_communities))\
                .replace("$topOutsideRels", str(self.top_outside_rels))\
                .replace("$topInsideRels", str(self.top_inside_rels))

            # 初始化向量存储
            vector_store = Neo4jVector.from_existing_index(
                self.embeddings,
                url=self.neo4j_uri,
                username=self.neo4j_username,
                password=self.neo4j_password,
                index_name=self.index_name,
                retrieval_query=final_query
            )

            # 执行相似度搜索
            docs = vector_store.similarity_search(
                query,
                k=self.top_entities
            )
            
            # 提取上下文内容
            if docs:
                # 处理不同的文档格式
                if hasattr(docs[0], 'page_content'):
                    context = docs[0].page_content
                elif isinstance(docs[0], dict):
                    context = docs[0].get('page_content', str(docs[0]))
                else:
                    context = str(docs[0])
            else:
                context = ""
            
            # 使用LLM生成响应
            response = chain.invoke({
                "context": context,
                "input": query,
                "response_type": self.response_type
            })
            
            # 验证搜索结果
            if not SearchUtils.validate_search_result(response):
                return "抱歉，未能找到相关信息来回答您的问题。"
            
            return response
            
        except Exception as e:
            print(f"[LocalSearch] 搜索失败: {e}")
            return f"搜索过程中出现问题: {str(e)}"
    
    def search_with_entities(self, query: str, entity_ids: List[str] = None) -> str:
        """
        基于指定实体进行本地搜索
        
        参数:
            query: 搜索查询字符串
            entity_ids: 指定的实体ID列表
            
        返回:
            str: 生成的最终答案
        """
        if not entity_ids:
            return self.search(query)
        
        try:
            # 获取实体相关的上下文信息
            db_manager = get_db_manager()
            
            # 构建基于实体的检索查询
            entity_query = """
            MATCH (e:__Entity__)
            WHERE e.id IN $entity_ids
            WITH collect(e) as entities
            """ + self.retrieval_query.replace("collect(node) as nodes", "entities as nodes")
            
            result = db_manager.execute_query(entity_query, {
                "entity_ids": entity_ids,
                "topChunks": self.top_chunks,
                "topCommunities": self.top_communities,
                "topOutsideRels": self.top_outside_rels,
                "topInsideRels": self.top_inside_rels,
            })
            
            if result.empty:
                return self.search(query)  # 回退到常规搜索
            
            # 提取上下文
            context_data = result.iloc[0]['text'] if 'text' in result.columns else ""

            # 格式化上下文
            if isinstance(context_data, dict):
                chunks = context_data.get('Chunks', [])
                reports = context_data.get('Reports', [])
                relationships = context_data.get('Relationships', [])
                entities = context_data.get('Entities', [])

                # 安全地处理chunks
                chunk_texts = []
                if chunks:
                    for chunk in chunks:
                        if isinstance(chunk, dict):
                            chunk_texts.append(chunk.get('text', ''))
                        else:
                            chunk_texts.append(str(chunk))

                # 安全地处理entities
                entity_texts = []
                if entities:
                    for entity in entities:
                        if isinstance(entity, dict):
                            entity_texts.append(entity.get('descriptionText', ''))
                        else:
                            entity_texts.append(str(entity))

                # 安全地处理relationships
                rel_texts = []
                if relationships:
                    for rel in relationships:
                        if isinstance(rel, dict):
                            rel_texts.append(rel.get('descriptionText', ''))
                        else:
                            rel_texts.append(str(rel))

                context = SearchUtils.format_search_context(
                    chunk_texts,
                    entity_texts,
                    rel_texts,
                    reports if isinstance(reports, list) else [str(reports)] if reports else []
                )
            else:
                context = str(context_data)
            
            # 使用LLM生成响应
            prompt = ChatPromptTemplate.from_messages([
                ("system", LC_SYSTEM_PROMPT),
                ("human", """
                    ---分析报告--- 
                    {context}

                    用户的问题是：
                    {input}

                    请按以下格式输出回答：
                    1. 使用三级标题(###)标记主题
                    2. 主要内容用清晰的段落展示
                    3. 最后必须用"#### 引用数据"标记引用部分，列出用到的数据来源
                    """
                 )
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            
            response = chain.invoke({
                "context": context,
                "input": query,
                "response_type": self.response_type
            })
            
            # 验证搜索结果
            if not SearchUtils.validate_search_result(response):
                return "抱歉，未能找到相关信息来回答您的问题。"
            
            return response
            
        except Exception as e:
            print(f"[LocalSearch] 基于实体的搜索失败: {e}")
            return self.search(query)  # 回退到常规搜索
    
    def get_search_context(self, query: str) -> Dict[str, Any]:
        """
        获取搜索上下文信息（不生成最终答案）
        
        参数:
            query: 搜索查询字符串
            
        返回:
            Dict[str, Any]: 搜索上下文信息
        """
        try:
            # 初始化向量存储
            vector_store = Neo4jVector.from_existing_index(
                self.embeddings,
                url=self.neo4j_uri,
                username=self.neo4j_username,
                password=self.neo4j_password,
                index_name=self.index_name,
                retrieval_query=self.retrieval_query
            )
            
            # 执行相似度搜索
            docs = vector_store.similarity_search(
                query,
                k=self.top_entities,
                params={
                    "topChunks": self.top_chunks,
                    "topCommunities": self.top_communities,
                    "topOutsideRels": self.top_outside_rels,
                    "topInsideRels": self.top_inside_rels,
                }
            )
            
            # 返回上下文信息
            return {
                "query": query,
                "documents": docs,
                "context": docs[0].page_content if docs else "",
                "metadata": docs[0].metadata if docs else {}
            }
            
        except Exception as e:
            print(f"[LocalSearch] 获取搜索上下文失败: {e}")
            return {
                "query": query,
                "documents": [],
                "context": "",
                "metadata": {},
                "error": str(e)
            }
    
    def close(self):
        """关闭Neo4j驱动连接"""
        # 连接由数据库管理器管理，这里不需要手动关闭
        pass
        
    def __enter__(self):
        """上下文管理器入口"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

import numpy as np
from typing import List, Dict, Any, Union

class VectorUtils:
    """向量搜索和相似度计算的统一工具类"""
    
    @staticmethod
    def cosine_similarity(vec1: Union[List[float], np.ndarray], 
                         vec2: Union[List[float], np.ndarray]) -> float:
        """
        计算两个向量的余弦相似度
        
        参数:
            vec1: 第一个向量
            vec2: 第二个向量
            
        返回:
            float: 相似度值 (0-1)
        """
        # 确保向量是numpy数组
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)
            
        # 计算余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        
        # 避免被零除
        if norm_a == 0 or norm_b == 0:
            return 0
            
        return dot_product / (norm_a * norm_b)
    
    @staticmethod
    def rank_by_similarity(query_embedding: List[float], 
                          candidates: List[Dict[str, Any]], 
                          embedding_field: str = "embedding",
                          top_k: int = None) -> List[Dict[str, Any]]:
        """
        对候选项按与查询向量的相似度排序
        
        参数:
            query_embedding: 查询向量
            candidates: 候选项列表，每项都包含embedding_field指定的字段
            embedding_field: 包含嵌入向量的字段名
            top_k: 返回的最大结果数，None表示返回所有结果
            
        返回:
            按相似度排序的候选项列表，每项增加"score"字段表示相似度
        """
        scored_items = []
        
        for item in candidates:
            if embedding_field in item and item[embedding_field]:
                # 计算相似度
                similarity = VectorUtils.cosine_similarity(query_embedding, item[embedding_field])
                # 复制item并添加分数
                scored_item = item.copy()
                scored_item["score"] = similarity
                scored_items.append(scored_item)
        
        # 按相似度降序排序
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        
        # 返回top_k结果
        if top_k:
            return scored_items[:top_k]
        return scored_items
    
    @staticmethod
    def filter_documents_by_relevance(query_embedding: List[float],
                                     docs: List, 
                                     embedding_attr: str = "embedding",
                                     threshold: float = 0.0,
                                     top_k: int = None) -> List:
        """
        基于相似度过滤文档
        
        参数:
            query_embedding: 查询向量
            docs: 文档列表，可以是具有embedding属性的对象
            embedding_attr: 嵌入向量的属性名称
            threshold: 最小相似度阈值
            top_k: 返回的最大结果数
            
        返回:
            按相似度排序的文档列表
        """
        scored_docs = []
        
        for doc in docs:
            # 获取文档的向量表示
            doc_embedding = getattr(doc, embedding_attr, None) if hasattr(doc, embedding_attr) else None
            
            if doc_embedding:
                similarity = VectorUtils.cosine_similarity(query_embedding, doc_embedding)
                # 只添加超过阈值的文档
                if similarity >= threshold:
                    scored_docs.append({
                        'document': doc,
                        'score': similarity
                    })
            else:
                # 如果没有向量，给一个基础分数
                scored_docs.append({
                    'document': doc,
                    'score': 0.0
                })
        
        # 按相似度降序排序
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        # 返回top_k结果
        if top_k:
            return scored_docs[:top_k]
        return scored_docs
    
    @staticmethod
    def batch_cosine_similarity(query_embedding: List[float],
                               embeddings_matrix: np.ndarray) -> np.ndarray:
        """
        批量计算查询向量与多个向量的余弦相似度
        
        参数:
            query_embedding: 查询向量
            embeddings_matrix: 嵌入向量矩阵，每行是一个向量
            
        返回:
            np.ndarray: 相似度数组
        """
        query_vec = np.array(query_embedding)
        
        # 计算点积
        dot_products = np.dot(embeddings_matrix, query_vec)
        
        # 计算范数
        query_norm = np.linalg.norm(query_vec)
        matrix_norms = np.linalg.norm(embeddings_matrix, axis=1)
        
        # 避免被零除
        norms_product = query_norm * matrix_norms
        norms_product[norms_product == 0] = 1e-8
        
        # 计算余弦相似度
        similarities = dot_products / norms_product
        
        return similarities
    
    @staticmethod
    def normalize_vector(vector: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        向量归一化
        
        参数:
            vector: 输入向量
            
        返回:
            np.ndarray: 归一化后的向量
        """
        vec = np.array(vector)
        norm = np.linalg.norm(vec)
        
        if norm == 0:
            return vec
        
        return vec / norm
    
    @staticmethod
    def euclidean_distance(vec1: Union[List[float], np.ndarray], 
                          vec2: Union[List[float], np.ndarray]) -> float:
        """
        计算两个向量的欧几里得距离
        
        参数:
            vec1: 第一个向量
            vec2: 第二个向量
            
        返回:
            float: 欧几里得距离
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        return np.linalg.norm(vec1 - vec2)
    
    @staticmethod
    def manhattan_distance(vec1: Union[List[float], np.ndarray], 
                          vec2: Union[List[float], np.ndarray]) -> float:
        """
        计算两个向量的曼哈顿距离
        
        参数:
            vec1: 第一个向量
            vec2: 第二个向量
            
        返回:
            float: 曼哈顿距离
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        return np.sum(np.abs(vec1 - vec2))

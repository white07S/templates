"""Reciprocal Rank Fusion (RRF) implementation for merging search results."""

from typing import List, Dict, Any, Tuple
from collections import defaultdict


def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[int, float, Dict[str, Any]]]],
    k: int = 60,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank_in_list))

    Args:
        result_lists: List of result lists, each containing (doc_id, score, document) tuples
        k: Constant for RRF formula (default 60)
        top_n: Number of top results to return

    Returns:
        List of top N documents sorted by RRF score
    """
    # Calculate RRF scores
    rrf_scores = defaultdict(float)
    doc_map = {}  # doc_id -> document

    for result_list in result_lists:
        for rank, (doc_id, score, document) in enumerate(result_list, start=1):
            rrf_score = 1.0 / (k + rank)
            rrf_scores[doc_id] += rrf_score

            # Store document if not already stored
            if doc_id not in doc_map:
                doc_map[doc_id] = document

    # Sort by RRF score
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Return top N documents
    final_results = []
    for doc_id, rrf_score in sorted_results[:top_n]:
        document = doc_map[doc_id]
        if document:
            # Add RRF score to document
            document['_rrf_score'] = rrf_score
            final_results.append(document)

    return final_results


def merge_search_results(
    dataframe_results: List[Dict[str, Any]] = None,
    keyword_results: List[Tuple[int, float, Dict[str, Any]]] = None,
    vector_results: List[Tuple[int, float, Dict[str, Any]]] = None,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """Merge results from different search modes.

    Args:
        dataframe_results: Results from dataframe search
        keyword_results: Results from keyword search
        vector_results: Results from vector search
        top_n: Number of top results to return

    Returns:
        Merged and ranked results
    """
    result_lists = []

    # Convert dataframe results to standard format
    if dataframe_results:
        df_list = [(i, 1.0, doc) for i, doc in enumerate(dataframe_results)]
        result_lists.append(df_list)

    # Add keyword results
    if keyword_results:
        result_lists.append(keyword_results)

    # Add vector results
    if vector_results:
        result_lists.append(vector_results)

    # If only one result list, return it directly
    if len(result_lists) == 1:
        results = []
        for item in result_lists[0][:top_n]:
            if isinstance(item, tuple) and len(item) == 3:
                results.append(item[2])  # Return just the document
            else:
                results.append(item)
        return results

    # Use RRF to merge multiple result lists
    return reciprocal_rank_fusion(result_lists, top_n=top_n)
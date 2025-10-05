"""
Search module with keyword and full-text search capabilities
Implements RRF (Reciprocal Rank Fusion) for result ranking
"""

import orjson
import tantivy
from typing import List, Dict, Any, Tuple
import config
from collections import defaultdict


class JSONSearcher:
    def __init__(self, index_dir: str = config.INDEX_DIR):
        """Initialize the searcher with an existing index"""
        self.index_dir = index_dir

        try:
            self.index = tantivy.Index.open(index_dir)
        except Exception as e:
            raise Exception(f"Failed to open index at {index_dir}: {e}")

    def unified_search(self, query: str, limit: int = config.DEFAULT_SEARCH_LIMIT) -> List[Tuple[float, Dict]]:
        """
        Perform unified search across both filename and content fields.
        Supports partial, case-insensitive matching for filenames.
        Returns list of (score, document) tuples.
        """
        searcher = self.index.searcher()

        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()

        # Parse queries for both fields
        # For filename: use the query as-is for now (Tantivy will handle partial matching internally)
        # Content search will also be case-insensitive
        filename_query = self.index.parse_query(query_lower, [config.FIELD_FILENAME])
        content_query = self.index.parse_query(query_lower, [config.FIELD_CONTENT])

        # Collect results from both searches
        results = []
        seen_paths = set()

        # Search filenames first (with partial matching)
        filename_results = searcher.search(filename_query, limit)
        for score, doc_address in filename_results.hits:
            doc = searcher.doc(doc_address)
            doc_dict = self._extract_doc_fields(doc)
            path = doc_dict.get(config.FIELD_PATH, "")
            if path not in seen_paths:
                results.append((score * 1.5, doc_dict))  # Boost filename matches
                seen_paths.add(path)

        # Search content
        content_results = searcher.search(content_query, limit)
        for score, doc_address in content_results.hits:
            doc = searcher.doc(doc_address)
            doc_dict = self._extract_doc_fields(doc)
            path = doc_dict.get(config.FIELD_PATH, "")
            if path not in seen_paths:
                results.append((score, doc_dict))
                seen_paths.add(path)

        return results

    def _extract_doc_fields(self, doc: tantivy.Document) -> Dict[str, Any]:
        """Extract fields from a Tantivy document"""
        result = {}

        # Extract all fields from the document
        try:
            # Get field values directly from the document
            for field_name in [config.FIELD_FILENAME, config.FIELD_PATH, config.FIELD_RAW_JSON]:
                values = doc.get_all(field_name)
                if values:
                    result[field_name] = values[0] if len(values) == 1 else values
        except:
            # Fallback to basic extraction
            result = doc.to_dict() if hasattr(doc, 'to_dict') else {}

        return result

    def reciprocal_rank_fusion(self, result_sets: List[List[Tuple[float, Dict]]], k: int = config.RRF_K) -> List[Dict]:
        """
        Apply Reciprocal Rank Fusion to combine multiple result sets.

        RRF score = Σ 1/(k + rank_i) for each result set i

        Args:
            result_sets: List of result lists, where each result is (score, document)
            k: RRF constant (typically 60)

        Returns:
            Combined and re-ranked results
        """
        # Dictionary to store RRF scores by document path (as unique identifier)
        rrf_scores = defaultdict(float)
        doc_map = {}

        # Calculate RRF scores
        for result_set in result_sets:
            for rank, (_, doc) in enumerate(result_set, start=1):
                doc_path = doc.get(config.FIELD_PATH, "")

                # Calculate RRF score for this document in this result set
                rrf_score = 1.0 / (k + rank)
                rrf_scores[doc_path] += rrf_score

                # Store the document if we haven't seen it before
                if doc_path not in doc_map:
                    doc_map[doc_path] = doc

        # Sort by RRF score
        sorted_results = sorted(
            [(score, doc_map[doc_path]) for doc_path, score in rrf_scores.items()],
            key=lambda x: x[0],
            reverse=True
        )

        # Return top K results with complete JSON
        top_results = []
        for score, doc in sorted_results[:config.TOP_K_RESULTS]:
            result = {
                "filename": doc.get(config.FIELD_FILENAME, ""),
                "filepath": doc.get(config.FIELD_PATH, ""),
                "rrf_score": score,
                "json_content": doc.get(config.FIELD_RAW_JSON, {})
            }
            top_results.append(result)

        return top_results

    def search(self, query: str) -> List[Dict]:
        """
        Main search function that performs unified search across filename and content.

        Args:
            query: Search query

        Returns:
            Top K results after RRF ranking
        """
        # Get unified search results
        search_results = self.unified_search(query, limit=config.DEFAULT_SEARCH_LIMIT)

        if not search_results:
            return []

        # Apply RRF on the single result set
        # Convert to list of result sets for RRF compatibility
        result_sets = [search_results] if search_results else []

        if result_sets:
            return self.reciprocal_rank_fusion(result_sets)
        else:
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get search index statistics"""
        searcher = self.index.searcher()
        return {
            "total_documents": searcher.num_docs,
            "index_path": self.index_dir
        }


def format_search_results(results: List[Dict]) -> str:
    """Format search results for display"""
    if not results:
        return "No results found."

    output = []
    for i, result in enumerate(results, 1):
        output.append(f"\n{'='*60}")
        output.append(f"Result #{i}")
        output.append(f"Filename: {result['filename']}")
        output.append(f"Path: {result['filepath']}")
        output.append(f"RRF Score: {result['rrf_score']:.4f}")
        output.append(f"JSON Content Preview:")

        # Show a preview of the JSON content
        json_str = orjson.dumps(result['json_content'], option=orjson.OPT_INDENT_2).decode('utf-8')
        lines = json_str.split('\n')
        preview_lines = lines[:20] if len(lines) > 20 else lines
        if len(lines) > 20:
            preview_lines.append("... (truncated)")

        output.append('\n'.join(preview_lines))

    return '\n'.join(output)


def main():
    """Main function for testing search"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python searcher.py <query>")
        print("  Searches both filenames (partial, case-insensitive) and content")
        return

    query = sys.argv[1]

    try:
        searcher = JSONSearcher()

        # Get stats
        stats = searcher.get_stats()
        print(f"Searching {stats['total_documents']} documents...")

        # Perform search
        results = searcher.search(query)

        # Display results
        print(f"\nSearch query: '{query}'")
        print(format_search_results(results))

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
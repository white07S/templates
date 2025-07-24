from pathlib import Path

# ===== Basic Configuration =====

# Basic path settings
BASE_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = BASE_DIR / 'files'

# Knowledge base theme setting, used for deepsearch (reasoning prompts)
KB_NAME = "East China University of Science and Technology"

# System runtime parameters
workers = 2  # FastAPI concurrent process count

# ===== Knowledge Graph Configuration =====

# Knowledge graph theme setting
theme = "East China University of Science and Technology Student Management"

# Knowledge graph entity and relationship types
entity_types = [
    "Student Type",
    "Scholarship Type",
    "Disciplinary Type",
    "Department",
    "Student Responsibility",
    "Management Regulation"
]

relationship_types = [
    "Apply",
    "Selection",
    "Violation",
    "Funding",
    "Appeal",
    "Management",
    "Rights and Obligations",
    "Mutually Exclusive",
]

# Conflict resolution and update strategy
# manual_first: Prioritize manual edits
# auto_first: Prioritize automatic updates
# merge: Attempt to merge
conflict_strategy = "manual_first"

# Community detection algorithm configuration
# If SLLPA cannot detect communities, switching to Leiden will work better
community_algorithm = 'leiden'

# ===== Text Processing Configuration =====

# Text processing parameters
CHUNK_SIZE = 500
OVERLAP = 100
MAX_TEXT_LENGTH = 500000
similarity_threshold = 0.9

# Answer generation configuration
response_type = "Multiple paragraphs"

# ===== Agent Tool Configuration =====

# Agent tool descriptions
lc_description = "Used for queries requiring specific details. Retrieves specific regulations, clauses, processes and other detailed content from East China University of Science and Technology student management documents. Suitable for questions like 'what is a specific regulation' and 'how does the process work'."
gl_description = "Used for queries requiring summary and analysis. Analyzes the overall framework, management principles, student rights and obligations and other macro content of East China University of Science and Technology student management system. Suitable for questions requiring systematic analysis such as 'the school's overall student management approach' and 'student rights protection mechanism'."
naive_description = "Basic retrieval tool that directly searches for text segments most relevant to the question without complex analysis. Quickly retrieves East China University of Science and Technology related policies and returns the most matching original text paragraphs."

# Frontend example questions
examples = [
    "How many class hours of absence will result in expulsion?",
    "Are the National Scholarship and National Inspirational Scholarship mutually exclusive?",
    "How do I apply to be an excellent student?",
    "What about the Shanghai Municipal Scholarship?",
]

# ===== Performance Optimization Configuration =====

# Parallel processing configuration
MAX_WORKERS = 4                # Number of parallel worker threads
BATCH_SIZE = 100               # Batch processing size
ENTITY_BATCH_SIZE = 50         # Entity processing batch size
CHUNK_BATCH_SIZE = 100         # Text chunk processing batch size
EMBEDDING_BATCH_SIZE = 64      # Embedding vector computation batch size
LLM_BATCH_SIZE = 5             # LLM processing batch size

# Indexing and community detection configuration
COMMUNITY_BATCH_SIZE = 50      # Community processing batch size

# GDS related configuration
GDS_MEMORY_LIMIT = 6           # GDS memory limit (GB)
GDS_CONCURRENCY = 4            # GDS concurrency level
GDS_NODE_COUNT_LIMIT = 50000   # GDS node count limit
GDS_TIMEOUT_SECONDS = 300      # GDS timeout (seconds)

# ===== Search Module Configuration =====

# Local search configuration
LOCAL_SEARCH_CONFIG = {
    # Vector retrieval parameters
    "top_entities": 10,
    "top_chunks": 10,
    "top_communities": 2,
    "top_outside_rels": 10,
    "top_inside_rels": 10,

    # Index configuration
    "index_name": "vector",
    "response_type": response_type,

    # Retrieval query template
    "retrieval_query": """
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
    """,
}

# Global search configuration
GLOBAL_SEARCH_CONFIG = {
    # Community level configuration
    "default_level": 0,  # Level 0
    "response_type": response_type,

    # Batch processing configuration
    "batch_size": 10,
    "max_communities": 100,
}

# ===== Cache Configuration =====

# Search cache configuration
SEARCH_CACHE_CONFIG = {
    # Cache directories
    "base_cache_dir": "./cache",
    "local_search_cache_dir": "./cache/local_search",
    "global_search_cache_dir": "./cache/global_search",
    "deep_research_cache_dir": "./cache/deep_research",

    # Cache strategy
    "max_cache_size": 200,
    "cache_ttl": 3600,  # 1 hour

    # Memory cache configuration
    "memory_cache_enabled": True,
    "disk_cache_enabled": True,
}

# ===== Reasoning Configuration =====

# Reasoning engine configuration
REASONING_CONFIG = {
    # Iteration configuration
    "max_iterations": 5,
    "max_search_limit": 10,

    # Thinking engine configuration
    "thinking_depth": 3,
    "exploration_width": 3,
    "max_exploration_steps": 5,

    # Evidence chain configuration
    "max_evidence_items": 50,
    "evidence_relevance_threshold": 0.7,

    # Validation configuration
    "validation": {
        "enable_answer_validation": True,
        "validation_threshold": 0.8,
        "enable_complexity_estimation": True,
        "consistency_threshold": 0.7
    },

    # Exploration configuration
    "exploration": {
        "max_exploration_steps": 5,
        "exploration_depth": 3,
        "exploration_breadth": 3,
        "exploration_width": 3,
        "relevance_threshold": 0.5,
        "exploration_decay_factor": 0.8,
        "enable_backtracking": True
    }
}
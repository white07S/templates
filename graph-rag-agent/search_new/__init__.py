# 核心搜索类
from search_new.core.local_search import LocalSearch
from search_new.core.global_search import GlobalSearch

# 搜索工具类
from search_new.tools.local_tool import LocalSearchTool
from search_new.tools.global_tool import GlobalSearchTool
from search_new.tools.hybrid_tool import HybridSearchTool
from search_new.tools.naive_search_tool import NaiveSearchTool
from search_new.tools.deep_research_tool import DeepResearchTool
from search_new.tools.deeper_research_tool import DeeperResearchTool

# 推理组件
from search_new.reasoning.engines.thinking_engine import ThinkingEngine
from search_new.reasoning.engines.search_engine import DualPathSearcher, QueryGenerator
from search_new.reasoning.engines.validator import AnswerValidator
from search_new.reasoning.enhancers.community_enhancer import CommunityAwareSearchEnhancer

# 工具类
from search_new.utils.vector_utils import VectorUtils
from search_new.utils.search_utils import SearchUtils
from search_new.reasoning.nlp.text_processor import TextProcessor
from search_new.reasoning.prompts.prompt_manager import PromptManager

# 配置
from search_new.config.search_config import SearchConfig, search_config

__version__ = "2.0.0"

__all__ = [
    # 核心搜索类
    "LocalSearch",
    "GlobalSearch",
    
    # 搜索工具类
    "LocalSearchTool",
    "GlobalSearchTool",
    "HybridSearchTool",
    "NaiveSearchTool",
    "DeepResearchTool",
    "DeeperResearchTool",
    
    # 推理组件
    "ThinkingEngine",
    "DualPathSearcher",
    "QueryGenerator", 
    "AnswerValidator",
    "CommunityAwareSearchEnhancer",
    
    # 工具类
    "VectorUtils",
    "SearchUtils",
    "TextProcessor",
    "PromptManager",
    
    # 配置
    "SearchConfig",
    "search_config"
]

# Community Detection and Summary Module

## File Structure

```
community/
├── __init__.py                    # Module entry point, exports factory classes
├── readme.md                      # Module documentation
├── detector/                      # Community detector directory
│   ├── __init__.py                # Detector factory class
│   ├── base.py                    # Basic detector abstract class
│   ├── leiden.py                  # Leiden algorithm implementation
│   ├── projections.py             # Graph projection mixin class
│   └── sllpa.py                   # SLLPA algorithm implementation
└── summary/                       # Community summary directory
    ├── __init__.py                # Summary factory class
    ├── base.py                    # Basic summary generator abstract class
    ├── leiden.py                  # Leiden community summary implementation
    └── sllpa.py                   # SLLPA community summary implementation
```

## Module Overview

This module provides support for community detection and summary functionality based on Neo4j graph databases, and is an important component of the knowledge graph project. Main features include:

1. Identifying community structures in graph data (community detection)
2. Generating summary descriptions for each community (community summaries)

## 设计思路与实现

### 设计模式

本模块采用多种设计模式确保代码的可维护性和可扩展性：

1. **工厂模式**：通过`CommunityDetectorFactory`和`CommunitySummarizerFactory`创建不同类型的检测器和摘要生成器，隐藏实现细节
2. **混入类（Mixin）**：使用`GraphProjectionMixin`提供共享的图投影功能
3. **上下文管理器**：在`BaseCommunityDetector`中使用`_graph_projection_context`管理资源生命周期
4. **模板方法模式**：在基类中定义算法骨架，由子类实现具体步骤

### 核心组件与流程

#### 1. 社区检测

**核心类**: `BaseCommunityDetector`  
**实现算法**: Leiden算法 (`LeidenDetector`) 和 SLLPA算法 (`SLLPADetector`)

**关键流程**:
1. **图投影**：通过`create_projection()`将Neo4j图数据投影到GDS库
2. **社区检测**：执行`detect_communities()`用特定算法识别社区结构
3. **结果保存**：通过`save_communities()`将社区信息持久化到图数据库
4. **资源清理**：使用`cleanup()`释放投影占用的资源

**自适应优化**:
- 根据系统资源（内存、CPU）自动调整算法参数
- 包含多层错误处理和备用方案
- 性能监控与统计

#### 2. 社区摘要

**核心类**: `BaseSummarizer`  
**辅助类**: `BaseCommunityDescriber`, `BaseCommunityRanker`, `BaseCommunityStorer`

**关键流程**:
1. **社区排名**：通过`calculate_ranks()`计算社区重要性排名
2. **信息收集**：通过`collect_community_info()`获取社区内节点和关系信息
3. **摘要生成**：使用LLM模型生成社区内容的语义摘要
4. **结果存储**：将摘要信息保存回图数据库

**性能优化**:
- 并行处理：利用`ThreadPoolExecutor`多线程生成摘要
- 分批处理：对大规模社区数据分批获取和处理
- 性能统计和监控

## 核心函数

### 社区检测模块

- **`BaseCommunityDetector.process()`**: 执行完整的社区检测流程，包括投影、检测和保存
  ```python
  def process(self) -> Dict[str, Any]:
      """执行完整的社区检测流程"""
      # 实现包括图投影、社区检测、结果保存和性能统计
  ```

- **`GraphProjectionMixin.create_projection()`**: 创建图投影，含多种降级策略
  ```python
  def create_projection(self) -> Tuple[Any, Dict]:
      """创建图投影，支持标准、过滤和保守多种模式"""
  ```

- **`LeidenDetector.detect_communities()`**: 执行Leiden算法社区检测
  ```python
  def detect_communities(self) -> Dict[str, Any]:
      """执行Leiden算法社区检测，含参数优化和失败降级"""
  ```

### 社区摘要模块

- **`BaseSummarizer.process_communities()`**: 处理所有社区的摘要生成流程
  ```python
  def process_communities(self) -> List[Dict]:
      """处理所有社区，包括权重计算、信息收集、摘要生成和存储"""
  ```

- **`BaseSummarizer._process_communities_parallel()`**: 并行处理社区摘要
  ```python
  def _process_communities_parallel(self, community_info: List[Dict], workers: int) -> List[Dict]:
      """利用多线程并行生成社区摘要"""
  ```

- **`LeidenSummarizer.collect_community_info()`**: 收集Leiden社区信息
  ```python
  def collect_community_info(self) -> List[Dict]:
      """收集社区信息，支持大规模批量处理"""
  ```

## 使用示例

### 社区检测

```python
from langchain_community.graphs import Neo4jGraph
from graphdatascience import GraphDataScience
from community import CommunityDetectorFactory

# 初始化图连接
graph = Neo4jGraph(url="neo4j://localhost:7687", username="neo4j", password="password")
gds = GraphDataScience("bolt://localhost:7687", auth=("neo4j", "password"))

# 创建社区检测器（可选算法：'leiden'或'sllpa'）
detector = CommunityDetectorFactory.create('leiden', gds, graph)

# 执行社区检测
results = detector.process()
print(f"社区检测结果: {results}")
```

### 社区摘要生成

```python
from community import CommunitySummarizerFactory

# 创建对应的摘要生成器
summarizer = CommunitySummarizerFactory.create_summarizer('leiden', graph)

# 生成社区摘要
summaries = summarizer.process_communities()
print(f"已生成 {len(summaries)} 个社区摘要")
```

## 性能考量

- 内存使用量与图大小成正比，为大图分析提供多级降级策略
- 社区摘要生成通过多线程并行处理提高效率
- 自适应系统资源，自动调整并发度和算法参数
- 完善的错误处理和监控，提供详细的性能统计

## 扩展性

- 通过继承`BaseCommunityDetector`添加新的社区检测算法
- 通过继承`BaseSummarizer`实现自定义摘要生成逻辑
- 工厂类支持轻松注册和使用新的实现
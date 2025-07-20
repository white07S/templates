import React, { useState, useEffect, useRef, useMemo } from 'react';
import * as echarts from 'echarts';
import 'echarts-gl';
import { Upload, Search, Filter, Download, Layers, Settings, Info, ZoomIn, ZoomOut, Maximize2, Move, GitBranch, Database, FileText, BarChart, Box, Circle, Network, Route } from 'lucide-react';

const LightRAGVisualizer = () => {
  const chartRef = useRef(null);
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [jsonAttributes, setJsonAttributes] = useState({});
  const [selectedNode, setSelectedNode] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [layoutType, setLayoutType] = useState('force');
  const [showLabels, setShowLabels] = useState(true);
  const [nodeSize, setNodeSize] = useState(30);
  const [activeTab, setActiveTab] = useState('graph');
  const [statistics, setStatistics] = useState({});
  const [communities, setCommunities] = useState({});
  const [chart, setChart] = useState(null);
  const [is3D, setIs3D] = useState(false);
  const [clusteringLevel, setClusteringLevel] = useState(0);
  const [pathfindingMode, setPathfindingMode] = useState(false);
  const [pathSource, setPathSource] = useState(null);
  const [pathTarget, setPathTarget] = useState(null);
  const [shortestPath, setShortestPath] = useState([]);

  // Parse GraphML file with LightRAG structure
  const parseGraphML = (xmlText) => {
    const parser = new DOMParser();
    const xmlDoc = parser.parseFromString(xmlText, 'text/xml');
    
    const nodes = [];
    const edges = [];
    const nodeMap = new Map();
    
    // Get key definitions
    const keys = {};
    const keyElements = xmlDoc.getElementsByTagName('key');
    Array.from(keyElements).forEach(key => {
      const id = key.getAttribute('id');
      const name = key.getAttribute('attr.name');
      keys[id] = name;
    });
    
    // Parse nodes
    const nodeElements = xmlDoc.getElementsByTagName('node');
    Array.from(nodeElements).forEach((node, index) => {
      const id = node.getAttribute('id');
      const attributes = {};
      
      // Extract data elements
      const dataElements = node.getElementsByTagName('data');
      Array.from(dataElements).forEach(data => {
        const key = data.getAttribute('key');
        const value = data.textContent;
        const attrName = keys[key] || key;
        attributes[attrName] = value;
      });
      
      const nodeData = {
        id,
        name: id,
        value: 10, // Default value
        category: attributes.entity_type || 'default',
        entityId: attributes.entity_id,
        description: attributes.description || '',
        sourceId: attributes.source_id,
        filePath: attributes.file_path,
        createdAt: attributes.created_at,
        attributes,
        // For force layout
        x: Math.random() * 1000 - 500,
        y: Math.random() * 1000 - 500,
        z: Math.random() * 1000 - 500
      };
      
      nodes.push(nodeData);
      nodeMap.set(id, index);
    });
    
    // Parse edges
    const edgeElements = xmlDoc.getElementsByTagName('edge');
    Array.from(edgeElements).forEach(edge => {
      const source = edge.getAttribute('source');
      const target = edge.getAttribute('target');
      const attributes = {};
      
      const dataElements = edge.getElementsByTagName('data');
      Array.from(dataElements).forEach(data => {
        const key = data.getAttribute('key');
        const value = data.textContent;
        const attrName = keys[key] || key;
        attributes[attrName] = value;
      });
      
      edges.push({
        source: nodeMap.get(source),
        target: nodeMap.get(target),
        value: parseFloat(attributes.weight || '1'),
        label: attributes.description || '',
        keywords: attributes.keywords || '',
        sourceId: attributes.source_id,
        filePath: attributes.file_path,
        createdAt: attributes.created_at,
        attributes
      });
    });
    
    // Update node values based on degree
    const degrees = new Array(nodes.length).fill(0);
    edges.forEach(edge => {
      degrees[edge.source]++;
      degrees[edge.target]++;
    });
    
    nodes.forEach((node, index) => {
      node.value = degrees[index] + 1;
    });
    
    return { nodes, edges };
  };

  // Dijkstra's algorithm for shortest path
  const findShortestPath = (nodes, edges, sourceIdx, targetIdx) => {
    const n = nodes.length;
    const dist = new Array(n).fill(Infinity);
    const prev = new Array(n).fill(null);
    const visited = new Array(n).fill(false);
    
    // Build adjacency list
    const adj = Array(n).fill(null).map(() => []);
    edges.forEach(edge => {
      adj[edge.source].push({ node: edge.target, weight: 1 / edge.value });
      adj[edge.target].push({ node: edge.source, weight: 1 / edge.value });
    });
    
    dist[sourceIdx] = 0;
    
    for (let i = 0; i < n; i++) {
      let u = -1;
      for (let j = 0; j < n; j++) {
        if (!visited[j] && (u === -1 || dist[j] < dist[u])) {
          u = j;
        }
      }
      
      if (dist[u] === Infinity) break;
      visited[u] = true;
      
      for (const neighbor of adj[u]) {
        const alt = dist[u] + neighbor.weight;
        if (alt < dist[neighbor.node]) {
          dist[neighbor.node] = alt;
          prev[neighbor.node] = u;
        }
      }
    }
    
    // Reconstruct path
    const path = [];
    let curr = targetIdx;
    while (curr !== null) {
      path.unshift(curr);
      curr = prev[curr];
    }
    
    return path[0] === sourceIdx ? path : [];
  };

  // Hierarchical clustering using modularity optimization
  const performClustering = (nodes, edges, level) => {
    if (level === 0) return {};
    
    // Simple community detection based on modularity
    const communities = {};
    const n = nodes.length;
    
    // Initialize each node in its own community
    for (let i = 0; i < n; i++) {
      communities[i] = [i];
    }
    
    // Merge communities based on edge density
    const iterations = Math.floor(level / 20);
    for (let iter = 0; iter < iterations; iter++) {
      // Find best merge
      let bestMerge = null;
      let bestGain = 0;
      
      const communityIds = Object.keys(communities);
      for (let i = 0; i < communityIds.length; i++) {
        for (let j = i + 1; j < communityIds.length; j++) {
          const comm1 = communities[communityIds[i]];
          const comm2 = communities[communityIds[j]];
          
          // Count edges between communities
          let edgeCount = 0;
          edges.forEach(edge => {
            if ((comm1.includes(edge.source) && comm2.includes(edge.target)) ||
                (comm2.includes(edge.source) && comm1.includes(edge.target))) {
              edgeCount++;
            }
          });
          
          const gain = edgeCount / (comm1.length * comm2.length);
          if (gain > bestGain) {
            bestGain = gain;
            bestMerge = [communityIds[i], communityIds[j]];
          }
        }
      }
      
      // Perform merge
      if (bestMerge && bestGain > 0.1) {
        const [id1, id2] = bestMerge;
        communities[id1] = [...communities[id1], ...communities[id2]];
        delete communities[id2];
      } else {
        break;
      }
    }
    
    // Renumber communities
    const finalCommunities = {};
    Object.values(communities).forEach((members, idx) => {
      finalCommunities[idx] = members;
    });
    
    return finalCommunities;
  };

  // Calculate layout positions
  const calculateLayout = (nodes, edges, type) => {
    const n = nodes.length;
    
    switch (type) {
      case 'hierarchical':
        // Simple hierarchical layout based on node degree
        const levels = {};
        const degrees = new Array(n).fill(0);
        edges.forEach(edge => {
          degrees[edge.source]++;
          degrees[edge.target]++;
        });
        
        // Assign levels based on degree
        const sorted = nodes.map((node, idx) => ({ idx, degree: degrees[idx] }))
          .sort((a, b) => b.degree - a.degree);
        
        sorted.forEach((item, i) => {
          const level = Math.floor(i * 5 / n);
          if (!levels[level]) levels[level] = [];
          levels[level].push(item.idx);
        });
        
        // Position nodes
        Object.entries(levels).forEach(([level, nodeIndices]) => {
          const y = (parseInt(level) - 2.5) * 200;
          nodeIndices.forEach((idx, i) => {
            const x = (i - nodeIndices.length / 2) * 100;
            nodes[idx].x = x;
            nodes[idx].y = y;
            nodes[idx].z = 0;
          });
        });
        break;
        
      case 'radial':
        // Radial layout with most connected nodes at center
        const centerNode = nodes.reduce((max, node, idx) => 
          degrees[idx] > degrees[max] ? idx : max, 0);
        
        // BFS from center
        const visited = new Array(n).fill(false);
        const queue = [[centerNode, 0]];
        const layers = {};
        
        visited[centerNode] = true;
        
        while (queue.length > 0) {
          const [nodeIdx, layer] = queue.shift();
          if (!layers[layer]) layers[layer] = [];
          layers[layer].push(nodeIdx);
          
          edges.forEach(edge => {
            let neighbor = -1;
            if (edge.source === nodeIdx && !visited[edge.target]) {
              neighbor = edge.target;
            } else if (edge.target === nodeIdx && !visited[edge.source]) {
              neighbor = edge.source;
            }
            
            if (neighbor !== -1) {
              visited[neighbor] = true;
              queue.push([neighbor, layer + 1]);
            }
          });
        }
        
        // Position nodes in circles
        Object.entries(layers).forEach(([layer, nodeIndices]) => {
          const radius = parseInt(layer) * 150;
          nodeIndices.forEach((idx, i) => {
            const angle = (i / nodeIndices.length) * 2 * Math.PI;
            nodes[idx].x = radius * Math.cos(angle);
            nodes[idx].y = radius * Math.sin(angle);
            nodes[idx].z = 0;
          });
        });
        break;
        
      case 'circular':
        // Simple circular layout
        nodes.forEach((node, i) => {
          const angle = (i / n) * 2 * Math.PI;
          const radius = 300;
          node.x = radius * Math.cos(angle);
          node.y = radius * Math.sin(angle);
          node.z = 0;
        });
        break;
    }
    
    return nodes;
  };

  // Initialize ECharts
  useEffect(() => {
    if (chartRef.current && graphData.nodes.length > 0) {
      const chartInstance = echarts.init(chartRef.current, null, {
        renderer: is3D ? 'canvas' : 'canvas',
        useDirtyRect: false
      });
      setChart(chartInstance);
      
      return () => {
        chartInstance.dispose();
      };
    }
  }, [graphData, is3D]);

  // Update chart when data or settings change
  useEffect(() => {
    if (chart && graphData.nodes.length > 0) {
      // Apply clustering
      const clusteredCommunities = performClustering(graphData.nodes, graphData.edges, clusteringLevel);
      setCommunities(clusteredCommunities);
      
      const stats = calculateStatistics(graphData.nodes, graphData.edges);
      setStatistics(stats);
      
      // Color scheme
      const colors = [
        '#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de',
        '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc', '#ff9f7f',
        '#37a2da', '#32c5e9', '#67e0e3', '#9fe6b8', '#ffdb5c'
      ];
      
      // Calculate layout if not force
      let layoutNodes = [...graphData.nodes];
      if (layoutType !== 'force' && layoutType !== 'none') {
        layoutNodes = calculateLayout(layoutNodes, graphData.edges, layoutType);
      }
      
      // Apply clustering colors
      const nodesWithCommunity = layoutNodes.map((node, index) => {
        let communityId = Object.keys(clusteredCommunities).find(id => 
          clusteredCommunities[id].includes(index)
        );
        
        // If no clustering, use entity type for coloring
        if (!communityId || clusteringLevel === 0) {
          const entityTypes = [...new Set(graphData.nodes.map(n => n.category))];
          communityId = entityTypes.indexOf(node.category);
        }
        
        return {
          ...node,
          itemStyle: {
            color: colors[parseInt(communityId || '0') % colors.length]
          }
        };
      });
      
      // Filter nodes
      const filteredNodes = nodesWithCommunity.filter(node => 
        (filterType === 'all' || node.category === filterType) &&
        (searchTerm === '' || node.name.toLowerCase().includes(searchTerm.toLowerCase()))
      );
      
      // Highlight path if in pathfinding mode
      let highlightedEdges = graphData.edges;
      if (pathfindingMode && shortestPath.length > 1) {
        highlightedEdges = graphData.edges.map(edge => {
          const isInPath = shortestPath.some((nodeIdx, i) => {
            if (i < shortestPath.length - 1) {
              const nextIdx = shortestPath[i + 1];
              return (edge.source === nodeIdx && edge.target === nextIdx) ||
                     (edge.target === nodeIdx && edge.source === nextIdx);
            }
            return false;
          });
          
          return {
            ...edge,
            lineStyle: {
              width: isInPath ? 8 : Math.log(edge.value + 1) * 2,
              color: isInPath ? '#ff0000' : undefined,
              opacity: isInPath ? 1 : 0.3
            }
          };
        });
      }
      
      const option = {
        tooltip: {
          trigger: 'item',
          formatter: (params) => {
            if (params.dataType === 'node') {
              const attrs = params.data;
              return `<strong>${params.data.name}</strong><br/>
                      Type: ${params.data.category}<br/>
                      ${attrs.description ? `Description: ${attrs.description}<br/>` : ''}
                      Connections: ${params.data.value}`;
            } else {
              return `${params.data.label || 'Edge'}<br/>
                      Weight: ${params.data.value}<br/>
                      ${params.data.keywords ? `Keywords: ${params.data.keywords}` : ''}`;
            }
          }
        },
        legend: {
          show: true,
          orient: 'vertical',
          left: 'left',
          data: clusteringLevel > 0 
            ? Object.keys(clusteredCommunities).map(id => `Cluster ${id}`)
            : [...new Set(graphData.nodes.map(n => n.category))]
        },
        animationDuration: 1500,
        animationEasingUpdate: 'quinticInOut',
        series: [{
          type: is3D ? 'scatter3D' : 'graph',
          layout: layoutType === 'force' ? 'force' : 'none',
          coordinateSystem: is3D ? 'cartesian3D' : undefined,
          data: is3D ? filteredNodes.map(node => [node.x, node.y, node.z, node.value, node]) : filteredNodes,
          links: is3D ? undefined : highlightedEdges,
          edges: is3D ? highlightedEdges : undefined,
          roam: !is3D,
          draggable: !is3D,
          focusNodeAdjacency: !is3D,
          label: {
            show: showLabels,
            position: 'right',
            formatter: '{b}',
            fontSize: 12
          },
          labelLayout: {
            hideOverlap: true
          },
          lineStyle: {
            color: 'source',
            curveness: 0.3
          },
          force: layoutType === 'force' ? {
            repulsion: 200,
            gravity: 0.1,
            edgeLength: 150,
            layoutAnimation: true
          } : undefined,
          symbolSize: (val) => {
            if (is3D) {
              return Math.min(Math.sqrt(val[3]) * nodeSize / 5, 50);
            }
            return Math.min(Math.sqrt(val) * nodeSize / 5, 50);
          },
          emphasis: {
            focus: 'adjacency',
            lineStyle: {
              width: 10
            }
          }
        }],
        xAxis3D: is3D ? { type: 'value' } : undefined,
        yAxis3D: is3D ? { type: 'value' } : undefined,
        zAxis3D: is3D ? { type: 'value' } : undefined,
        grid3D: is3D ? {
          viewControl: {
            projection: 'perspective',
            autoRotate: false,
            damping: 0.5,
            rotateSensitivity: 1,
            zoomSensitivity: 1,
            panSensitivity: 1
          }
        } : undefined
      };
      
      chart.setOption(option);
      
      // Handle node click
      chart.off('click');
      chart.on('click', (params) => {
        if (params.dataType === 'node' || (is3D && params.value)) {
          const nodeData = is3D ? params.value[4] : params.data;
          
          if (pathfindingMode) {
            if (!pathSource) {
              setPathSource(params.dataIndex);
            } else if (!pathTarget && params.dataIndex !== pathSource) {
              setPathTarget(params.dataIndex);
              // Calculate shortest path
              const path = findShortestPath(graphData.nodes, graphData.edges, pathSource, params.dataIndex);
              setShortestPath(path);
            } else {
              // Reset
              setPathSource(params.dataIndex);
              setPathTarget(null);
              setShortestPath([]);
            }
          } else {
            setSelectedNode(nodeData);
          }
        }
      });
    }
  }, [chart, graphData, layoutType, showLabels, nodeSize, filterType, searchTerm, is3D, 
      clusteringLevel, pathfindingMode, pathSource, pathTarget, shortestPath]);

  // Calculate statistics
  const calculateStatistics = (nodes, edges) => {
    const degrees = new Array(nodes.length).fill(0);
    edges.forEach(edge => {
      degrees[edge.source]++;
      degrees[edge.target]++;
    });
    
    const avgDegree = degrees.reduce((a, b) => a + b, 0) / nodes.length;
    const maxDegree = Math.max(...degrees);
    const minDegree = Math.min(...degrees);
    
    const typeCount = {};
    nodes.forEach(node => {
      const type = node.category;
      typeCount[type] = (typeCount[type] || 0) + 1;
    });
    
    return {
      nodeCount: nodes.length,
      edgeCount: edges.length,
      avgDegree: avgDegree.toFixed(2),
      maxDegree,
      minDegree,
      density: (2 * edges.length / (nodes.length * (nodes.length - 1))).toFixed(4),
      typeCount,
      clusterCount: Object.keys(communities).length
    };
  };

  // File handlers
  const handleGraphMLUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const xmlText = event.target.result;
        const parsedData = parseGraphML(xmlText);
        setGraphData(parsedData);
        // Reset pathfinding when new graph is loaded
        setPathfindingMode(false);
        setPathSource(null);
        setPathTarget(null);
        setShortestPath([]);
      };
      reader.readAsText(file);
    }
  };

  const handleJSONUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          const jsonData = JSON.parse(event.target.result);
          setJsonAttributes(jsonData);
        } catch (error) {
          console.error('Invalid JSON file:', error);
        }
      };
      reader.readAsText(file);
    }
  };

  // Export functions
  const exportGraph = () => {
    if (chart) {
      const url = chart.getDataURL({
        type: 'png',
        pixelRatio: 2,
        backgroundColor: '#fff'
      });
      const a = document.createElement('a');
      a.href = url;
      a.download = 'lightrag-graph.png';
      a.click();
    }
  };

  const exportData = () => {
    const exportObj = {
      nodes: graphData.nodes,
      edges: graphData.edges,
      statistics,
      communities,
      metadata: {
        layout: layoutType,
        is3D,
        clusteringLevel
      }
    };
    const dataStr = JSON.stringify(exportObj, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'lightrag-data.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  // Data table component
  const DataTable = ({ data, type }) => {
    if (!data || data.length === 0) return <div className="text-gray-500 text-center p-4">No data loaded</div>;
    
    const columns = type === 'nodes' 
      ? ['ID', 'Name', 'Type', 'Description', 'Connections']
      : ['Source', 'Target', 'Weight', 'Keywords', 'Description'];
    
    return (
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {columns.map(col => (
                <th key={col} className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {data.map((item, index) => (
              <tr key={index} className="hover:bg-gray-50">
                {type === 'nodes' ? (
                  <>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{item.id}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{item.name}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{item.category}</td>
                    <td className="px-6 py-4 text-sm text-gray-900 max-w-xs truncate">{item.description}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{item.value}</td>
                  </>
                ) : (
                  <>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{graphData.nodes[item.source]?.name}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{graphData.nodes[item.target]?.name}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{item.value}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{item.keywords || '-'}</td>
                    <td className="px-6 py-4 text-sm text-gray-900 max-w-xs truncate">{item.label || '-'}</td>
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <div className="w-80 bg-white shadow-lg p-6 overflow-y-auto">
        <h1 className="text-2xl font-bold mb-6 text-gray-800">LightRAG Visualizer</h1>
        
        {/* File Upload */}
        <div className="mb-6">
          <h2 className="text-lg font-semibold mb-3 flex items-center">
            <Upload className="mr-2" size={20} />
            Upload Files
          </h2>
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                GraphML File
              </label>
              <input
                type="file"
                accept=".graphml,.xml"
                onChange={handleGraphMLUpload}
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                JSON Attributes (Optional)
              </label>
              <input
                type="file"
                accept=".json"
                onChange={handleJSONUpload}
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-green-50 file:text-green-700 hover:file:bg-green-100"
              />
            </div>
          </div>
        </div>

        {/* Search and Filter */}
        <div className="mb-6">
          <h2 className="text-lg font-semibold mb-3 flex items-center">
            <Search className="mr-2" size={20} />
            Search & Filter
          </h2>
          <div className="space-y-3">
            <input
              type="text"
              placeholder="Search nodes..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Types</option>
              {graphData.nodes.length > 0 && 
                [...new Set(graphData.nodes.map(n => n.category))].map(type => (
                  <option key={type} value={type}>{type}</option>
                ))
              }
            </select>
          </div>
        </div>

        {/* Layout Settings */}
        <div className="mb-6">
          <h2 className="text-lg font-semibold mb-3 flex items-center">
            <Settings className="mr-2" size={20} />
            Layout Settings
          </h2>
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Layout Type
              </label>
              <select
                value={layoutType}
                onChange={(e) => setLayoutType(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="force">Force-Directed</option>
                <option value="circular">Circular</option>
                <option value="hierarchical">Hierarchical</option>
                <option value="radial">Radial</option>
                <option value="none">None</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Node Size ({nodeSize})
              </label>
              <input
                type="range"
                min="10"
                max="100"
                value={nodeSize}
                onChange={(e) => setNodeSize(e.target.value)}
                className="w-full"
              />
            </div>
            <div className="flex items-center">
              <input
                type="checkbox"
                id="showLabels"
                checked={showLabels}
                onChange={(e) => setShowLabels(e.target.checked)}
                className="mr-2"
              />
              <label htmlFor="showLabels" className="text-sm font-medium text-gray-700">
                Show Labels
              </label>
            </div>
            <div className="flex items-center">
              <input
                type="checkbox"
                id="is3D"
                checked={is3D}
                onChange={(e) => setIs3D(e.target.checked)}
                className="mr-2"
              />
              <label htmlFor="is3D" className="text-sm font-medium text-gray-700">
                3D Visualization
              </label>
            </div>
          </div>
        </div>

        {/* Clustering Controls */}
        <div className="mb-6">
          <h2 className="text-lg font-semibold mb-3 flex items-center">
            <Network className="mr-2" size={20} />
            Clustering
          </h2>
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Clustering Level ({clusteringLevel})
              </label>
              <input
                type="range"
                min="0"
                max="100"
                value={clusteringLevel}
                onChange={(e) => setClusteringLevel(parseInt(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        </div>

        {/* Pathfinding */}
        <div className="mb-6">
          <h2 className="text-lg font-semibold mb-3 flex items-center">
            <Route className="mr-2" size={20} />
            Pathfinding
          </h2>
          <div className="space-y-3">
            <button
              onClick={() => {
                setPathfindingMode(!pathfindingMode);
                setPathSource(null);
                setPathTarget(null);
                setShortestPath([]);
              }}
              className={`w-full px-4 py-2 rounded-md transition-colors ${
                pathfindingMode 
                  ? 'bg-purple-600 text-white hover:bg-purple-700' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              {pathfindingMode ? 'Exit Pathfinding' : 'Enter Pathfinding'}
            </button>
            {pathfindingMode && (
              <div className="text-sm text-gray-600">
                {!pathSource && 'Click a node to set source'}
                {pathSource !== null && !pathTarget && 'Click another node to set target'}
                {pathSource !== null && pathTarget !== null && (
                  <div>
                    Path: {graphData.nodes[pathSource]?.name} → {graphData.nodes[pathTarget]?.name}
                    <br />
                    Length: {shortestPath.length - 1} steps
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Statistics */}
        {statistics.nodeCount && (
          <div className="mb-6">
            <h2 className="text-lg font-semibold mb-3 flex items-center">
              <BarChart className="mr-2" size={20} />
              Statistics
            </h2>
            <div className="bg-gray-50 p-3 rounded-md text-sm space-y-1">
              <div>Nodes: {statistics.nodeCount}</div>
              <div>Edges: {statistics.edgeCount}</div>
              <div>Avg Degree: {statistics.avgDegree}</div>
              <div>Density: {statistics.density}</div>
              <div>Clusters: {statistics.clusterCount}</div>
              <div className="mt-2">
                <div className="font-medium">Entity Types:</div>
                {Object.entries(statistics.typeCount).map(([type, count]) => (
                  <div key={type} className="ml-2">{type}: {count}</div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Export */}
        <div className="mb-6">
          <h2 className="text-lg font-semibold mb-3 flex items-center">
            <Download className="mr-2" size={20} />
            Export
          </h2>
          <div className="space-y-2">
            <button
              onClick={exportGraph}
              className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            >
              Export as Image
            </button>
            <button
              onClick={exportData}
              className="w-full px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
            >
              Export Data (JSON)
            </button>
          </div>
        </div>

        {/* Selected Node Info */}
        {selectedNode && (
          <div className="mb-6">
            <h2 className="text-lg font-semibold mb-3 flex items-center">
              <Info className="mr-2" size={20} />
              Selected Node
            </h2>
            <div className="bg-blue-50 p-3 rounded-md text-sm">
              <div className="font-semibold">{selectedNode.name}</div>
              <div className="mt-2 space-y-1">
                <div><span className="font-medium">Type:</span> {selectedNode.category}</div>
                {selectedNode.description && (
                  <div><span className="font-medium">Description:</span> {selectedNode.description}</div>
                )}
                <div><span className="font-medium">Connections:</span> {selectedNode.value}</div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Tabs */}
        <div className="bg-white border-b px-6 py-3">
          <div className="flex space-x-6">
            <button
              onClick={() => setActiveTab('graph')}
              className={`flex items-center px-3 py-2 rounded-md transition-colors ${
                activeTab === 'graph'
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              {is3D ? <Box className="mr-2" size={18} /> : <GitBranch className="mr-2" size={18} />}
              Graph Visualization
            </button>
            <button
              onClick={() => setActiveTab('nodes')}
              className={`flex items-center px-3 py-2 rounded-md transition-colors ${
                activeTab === 'nodes'
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Database className="mr-2" size={18} />
              Nodes Table
            </button>
            <button
              onClick={() => setActiveTab('edges')}
              className={`flex items-center px-3 py-2 rounded-md transition-colors ${
                activeTab === 'edges'
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Layers className="mr-2" size={18} />
              Edges Table
            </button>
            <button
              onClick={() => setActiveTab('json')}
              className={`flex items-center px-3 py-2 rounded-md transition-colors ${
                activeTab === 'json'
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <FileText className="mr-2" size={18} />
              JSON Attributes
            </button>
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 p-6 bg-gray-50">
          {activeTab === 'graph' && (
            <div className="h-full bg-white rounded-lg shadow-md p-4">
              {graphData.nodes.length > 0 ? (
                <div ref={chartRef} className="w-full h-full" />
              ) : (
                <div className="flex items-center justify-center h-full text-gray-500">
                  <div className="text-center">
                    <GitBranch size={48} className="mx-auto mb-4 text-gray-400" />
                    <p>Upload a GraphML file to visualize the graph</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'nodes' && (
            <div className="bg-white rounded-lg shadow-md overflow-hidden">
              <DataTable data={graphData.nodes} type="nodes" />
            </div>
          )}

          {activeTab === 'edges' && (
            <div className="bg-white rounded-lg shadow-md overflow-hidden">
              <DataTable data={graphData.edges} type="edges" />
            </div>
          )}

          {activeTab === 'json' && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <pre className="bg-gray-50 p-4 rounded overflow-auto max-h-96">
                {Object.keys(jsonAttributes).length > 0 
                  ? JSON.stringify(jsonAttributes, null, 2)
                  : 'No JSON attributes loaded'}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LightRAGVisualizer;

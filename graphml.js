import React, { useState, useEffect, useRef, useMemo } from 'react';
import * as echarts from 'echarts';
import { Upload, Search, Filter, Download, Layers, Settings, Info, ZoomIn, ZoomOut, Maximize2, Move, GitBranch, Database, FileText, BarChart } from 'lucide-react';

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

  // Parse GraphML file
  const parseGraphML = (xmlText) => {
    const parser = new DOMParser();
    const xmlDoc = parser.parseFromString(xmlText, 'text/xml');
    
    const nodes = [];
    const edges = [];
    const nodeMap = new Map();
    
    // Parse nodes
    const nodeElements = xmlDoc.getElementsByTagName('node');
    Array.from(nodeElements).forEach((node, index) => {
      const id = node.getAttribute('id');
      const dataElements = node.getElementsByTagName('data');
      const attributes = {};
      
      Array.from(dataElements).forEach(data => {
        const key = data.getAttribute('key');
        const value = data.textContent;
        attributes[key] = value;
      });
      
      const nodeData = {
        id,
        name: attributes.label || attributes.name || id,
        value: parseInt(attributes.weight || attributes.size || '10'),
        category: attributes.type || attributes.category || 'default',
        attributes,
        x: Math.random() * 1000,
        y: Math.random() * 1000
      };
      
      nodes.push(nodeData);
      nodeMap.set(id, index);
    });
    
    // Parse edges
    const edgeElements = xmlDoc.getElementsByTagName('edge');
    Array.from(edgeElements).forEach(edge => {
      const source = edge.getAttribute('source');
      const target = edge.getAttribute('target');
      const dataElements = edge.getElementsByTagName('data');
      const attributes = {};
      
      Array.from(dataElements).forEach(data => {
        const key = data.getAttribute('key');
        const value = data.textContent;
        attributes[key] = value;
      });
      
      edges.push({
        source: nodeMap.get(source),
        target: nodeMap.get(target),
        value: parseFloat(attributes.weight || '1'),
        label: attributes.label || attributes.description || '',
        attributes
      });
    });
    
    return { nodes, edges };
  };

  // Detect communities using a simple algorithm
  const detectCommunities = (nodes, edges) => {
    const communities = {};
    const visited = new Set();
    let communityId = 0;
    
    const dfs = (nodeIndex, communityId) => {
      if (visited.has(nodeIndex)) return;
      visited.add(nodeIndex);
      
      if (!communities[communityId]) {
        communities[communityId] = [];
      }
      communities[communityId].push(nodeIndex);
      
      edges.forEach(edge => {
        if (edge.source === nodeIndex && !visited.has(edge.target)) {
          dfs(edge.target, communityId);
        } else if (edge.target === nodeIndex && !visited.has(edge.source)) {
          dfs(edge.source, communityId);
        }
      });
    };
    
    nodes.forEach((node, index) => {
      if (!visited.has(index)) {
        dfs(index, communityId);
        communityId++;
      }
    });
    
    return communities;
  };

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
      typeCount
    };
  };

  // Initialize ECharts
  useEffect(() => {
    if (chartRef.current && graphData.nodes.length > 0) {
      const chartInstance = echarts.init(chartRef.current, null, {
        renderer: 'canvas',
        useDirtyRect: false
      });
      setChart(chartInstance);
      
      return () => {
        chartInstance.dispose();
      };
    }
  }, [graphData]);

  // Update chart when data or settings change
  useEffect(() => {
    if (chart && graphData.nodes.length > 0) {
      const communityData = detectCommunities(graphData.nodes, graphData.edges);
      setCommunities(communityData);
      
      const stats = calculateStatistics(graphData.nodes, graphData.edges);
      setStatistics(stats);
      
      // Assign community colors
      const colors = [
        '#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de',
        '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc', '#ff9f7f'
      ];
      
      const nodesWithCommunity = graphData.nodes.map((node, index) => {
        const communityId = Object.keys(communityData).find(id => 
          communityData[id].includes(index)
        );
        return {
          ...node,
          category: communityId || '0',
          itemStyle: {
            color: colors[parseInt(communityId || '0') % colors.length]
          }
        };
      });
      
      const option = {
        tooltip: {
          trigger: 'item',
          formatter: (params) => {
            if (params.dataType === 'node') {
              const attrs = params.data.attributes || {};
              let attrStr = Object.entries(attrs)
                .map(([k, v]) => `${k}: ${v}`)
                .join('<br/>');
              return `<strong>${params.data.name}</strong><br/>
                      Category: ${params.data.category}<br/>
                      Value: ${params.data.value}<br/>
                      ${attrStr}`;
            } else {
              return `${params.data.label || 'Edge'}<br/>
                      Weight: ${params.data.value}`;
            }
          }
        },
        legend: {
          show: true,
          orient: 'vertical',
          left: 'left',
          data: Object.keys(communityData).map(id => `Community ${id}`)
        },
        animationDuration: 1500,
        animationEasingUpdate: 'quinticInOut',
        series: [{
          type: 'graph',
          layout: layoutType,
          data: nodesWithCommunity.filter(node => 
            filterType === 'all' || node.category === filterType
          ).filter(node =>
            searchTerm === '' || 
            node.name.toLowerCase().includes(searchTerm.toLowerCase())
          ),
          links: graphData.edges.map(edge => ({
            ...edge,
            lineStyle: {
              width: Math.log(edge.value + 1) * 2,
              curveness: 0.3,
              opacity: 0.7
            }
          })),
          roam: true,
          draggable: true,
          focusNodeAdjacency: true,
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
          force: {
            repulsion: 100,
            gravity: 0.1,
            edgeLength: 100,
            layoutAnimation: true
          },
          symbolSize: (val) => Math.min(Math.sqrt(val) * nodeSize / 10, 100),
          emphasis: {
            focus: 'adjacency',
            lineStyle: {
              width: 10
            }
          }
        }]
      };
      
      chart.setOption(option);
      
      // Handle node click
      chart.on('click', (params) => {
        if (params.dataType === 'node') {
          setSelectedNode(params.data);
        }
      });
    }
  }, [chart, graphData, layoutType, showLabels, nodeSize, filterType, searchTerm]);

  // File handlers
  const handleGraphMLUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const xmlText = event.target.result;
        const parsedData = parseGraphML(xmlText);
        setGraphData(parsedData);
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
      communities
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
      ? ['ID', 'Name', 'Category', 'Value']
      : ['Source', 'Target', 'Label', 'Weight'];
    
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
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{item.value}</td>
                  </>
                ) : (
                  <>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{graphData.nodes[item.source]?.name}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{graphData.nodes[item.target]?.name}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{item.label || '-'}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{item.value}</td>
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
              <option value="all">All Communities</option>
              {Object.keys(communities).map(id => (
                <option key={id} value={id}>Community {id}</option>
              ))}
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
              <div>Communities: {Object.keys(communities).length}</div>
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
                {Object.entries(selectedNode.attributes || {}).map(([key, value]) => (
                  <div key={key} className="text-xs">
                    <span className="font-medium">{key}:</span> {value}
                  </div>
                ))}
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
              <GitBranch className="mr-2" size={18} />
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

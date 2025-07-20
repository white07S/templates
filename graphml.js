import React, { useState, useRef, useCallback } from 'react';
import ReactECharts from 'echarts-for-react';
import { convert } from 'xml-js';

const GraphMLVisualizer = () => {
  const [graphData, setGraphData] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [showNodeLabels, setShowNodeLabels] = useState(true);
  const [showEdgeLabels, setShowEdgeLabels] = useState(true);
  const fileInputRef = useRef(null);

  const parseGraphML = useCallback((xmlString) => {
    const jsonData = JSON.parse(convert(xmlString, { compact: true }));
    const graphml = jsonData.graphml;
    
    // Extract key definitions
    const keys = graphml.key.reduce((acc, key) => {
      acc[key._attributes.id] = key._attributes['attr.name'];
      return acc;
    }, {});
    
    // Parse nodes
    const nodes = graphml.graph.node.map(node => {
      const attributes = node.data.reduce((acc, data) => {
        const keyName = keys[data._attributes.key];
        acc[keyName] = data._text;
        return acc;
      }, {});
      
      return {
        id: node._attributes.id,
        name: attributes.entity_id,
        category: attributes.entity_type,
        description: attributes.description,
        symbolSize: 30,
        itemStyle: {
          color: getCategoryColor(attributes.entity_type)
        }
      };
    });
    
    // Create node index for quick lookups
    const nodeMap = nodes.reduce((acc, node) => {
      acc[node.id] = node;
      return acc;
    }, {});
    
    // Parse edges
    const edges = graphml.graph.edge.map(edge => {
      const attributes = Array.isArray(edge.data) 
        ? edge.data.reduce((acc, data) => {
            const keyName = keys[data._attributes.key];
            acc[keyName] = data._text;
            return acc;
          }, {})
        : {};
      
      return {
        source: edge._attributes.source,
        target: edge._attributes.target,
        name: attributes.description || '',
        value: parseFloat(attributes.weight) || 1,
        lineStyle: {
          width: attributes.weight ? parseFloat(attributes.weight) / 3 : 1
        }
      };
    });
    
    return { nodes, edges, nodeMap };
  }, []);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const graphData = parseGraphML(e.target.result);
        setGraphData(graphData);
        setSelectedNode(null);
      } catch (error) {
        console.error('Error parsing GraphML:', error);
        alert('Invalid GraphML file format');
      }
    };
    reader.readAsText(file);
  };

  const getCategoryColor = (category) => {
    const colors = {
      person: '#FF6B6B',
      organization: '#4ECDC4',
      category: '#FFD166',
      default: '#A0A0A0'
    };
    return colors[category] || colors.default;
  };

  const handleNodeClick = (params) => {
    if (!graphData || !params.data) return;
    setSelectedNode(params.data.id);
  };

  const getConnectedSubgraph = () => {
    if (!selectedNode || !graphData) return { nodes: [], edges: [] };
    
    const connectedNodes = new Set([selectedNode]);
    const connectedEdges = [];
    const queue = [selectedNode];
    
    while (queue.length > 0) {
      const current = queue.shift();
      graphData.edges.forEach(edge => {
        if (edge.source === current && !connectedNodes.has(edge.target)) {
          connectedNodes.add(edge.target);
          queue.push(edge.target);
          connectedEdges.push(edge);
        } else if (edge.target === current && !connectedNodes.has(edge.source)) {
          connectedNodes.add(edge.source);
          queue.push(edge.source);
          connectedEdges.push(edge);
        }
      });
    }
    
    return {
      nodes: graphData.nodes.filter(node => connectedNodes.has(node.id)),
      edges: connectedEdges
    };
  };

  const getChartOptions = (type) => {
    if (!graphData) return {};
    
    const displayData = selectedNode ? getConnectedSubgraph() : graphData;
    const { nodes, edges } = displayData;

    const baseOptions = {
      tooltip: {
        formatter: params => {
          if (params.dataType === 'node') {
            return `<b>${params.data.name}</b><br/>
                    Type: ${params.data.category}<br/>
                    ${params.data.description}`;
          }
          return `<b>${params.data.name || 'Connection'}</b><br/>
                  Weight: ${params.data.value}`;
        }
      },
      series: [{
        type: 'graph',
        layout: type === 'radial' ? 'circular' : type === 'matrix' ? 'none' : 'force',
        roam: true,
        focusNodeAdjacency: true,
        label: {
          show: showNodeLabels,
          position: 'right',
          formatter: '{b}'
        },
        edgeLabel: {
          show: showEdgeLabels,
          formatter: '{c}'
        },
        edgeSymbol: ['none', 'arrow'],
        edgeSymbolSize: [4, 10],
        lineStyle: {
          color: 'source',
          curveness: 0.1
        },
        emphasis: {
          focus: 'adjacency',
          lineStyle: {
            width: 5
          }
        }
      }]
    };

    switch (type) {
      case 'radial':
        return {
          ...baseOptions,
          series: [{
            ...baseOptions.series[0],
            circular: { rotateLabel: true },
            data: nodes,
            links: edges,
            categories: [
              { name: 'person' },
              { name: 'organization' },
              { name: 'category' }
            ]
          }]
        };

      case 'force':
        return {
          ...baseOptions,
          series: [{
            ...baseOptions.series[0],
            data: nodes,
            links: edges,
            force: {
              repulsion: 200,
              gravity: 0.1,
              edgeLength: 100
            }
          }]
        };

      case 'matrix':
        const matrixNodes = nodes.map(node => node.name);
        const matrixData = [];
        
        edges.forEach(edge => {
          matrixData.push([
            matrixNodes.indexOf(graphData.nodeMap[edge.source].name),
            matrixNodes.indexOf(graphData.nodeMap[edge.target].name),
            edge.value
          ]);
        });
        
        return {
          tooltip: {
            position: 'top',
            formatter: params => {
              const source = matrixNodes[params.data[0]];
              const target = matrixNodes[params.data[1]];
              return `<b>${source} → ${target}</b><br/>Weight: ${params.data[2]}`;
            }
          },
          grid: { top: 50, left: 120, right: 50, bottom: 120 },
          xAxis: {
            type: 'category',
            data: matrixNodes,
            position: 'top',
            axisLabel: { rotate: 45, interval: 0 }
          },
          yAxis: {
            type: 'category',
            data: matrixNodes,
            axisLabel: { interval: 0 }
          },
          visualMap: {
            min: 0,
            max: Math.max(...edges.map(e => e.value)),
            calculable: true,
            orient: 'horizontal',
            left: 'center',
            bottom: 10
          },
          series: [{
            name: 'Connections',
            type: 'heatmap',
            data: matrixData,
            emphasis: {
              itemStyle: {
                shadowBlur: 10,
                shadowColor: 'rgba(0, 0, 0, 0.5)'
              }
            }
          }]
        };

      default:
        return baseOptions;
    }
  };

  return (
    <div className="graphml-visualizer">
      <div className="controls">
        <input
          type="file"
          accept=".graphml"
          onChange={handleFileUpload}
          ref={fileInputRef}
          style={{ display: 'none' }}
        />
        <button onClick={() => fileInputRef.current.click()}>
          Upload GraphML File
        </button>
        
        {graphData && (
          <>
            <div className="toggle-group">
              <label>
                <input
                  type="checkbox"
                  checked={showNodeLabels}
                  onChange={() => setShowNodeLabels(!showNodeLabels)}
                />
                Node Labels
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={showEdgeLabels}
                  onChange={() => setShowEdgeLabels(!showEdgeLabels)}
                />
                Edge Labels
              </label>
            </div>
            
            {selectedNode && (
              <div className="node-info">
                <h3>{graphData.nodeMap[selectedNode].name}</h3>
                <p>{graphData.nodeMap[selectedNode].description}</p>
                <button onClick={() => setSelectedNode(null)}>
                  Reset View
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {graphData ? (
        <div className="charts-container">
          <div className="chart">
            <h3>Radial View</h3>
            <ReactECharts
              option={getChartOptions('radial')}
              style={{ height: '500px' }}
              onEvents={{ click: handleNodeClick }}
            />
          </div>
          
          <div className="chart">
            <h3>Force-Directed View</h3>
            <ReactECharts
              option={getChartOptions('force')}
              style={{ height: '500px' }}
              onEvents={{ click: handleNodeClick }}
            />
          </div>
          
          <div className="chart">
            <h3>Adjacency Matrix</h3>
            <ReactECharts
              option={getChartOptions('matrix')}
              style={{ height: '500px' }}
            />
          </div>
        </div>
      ) : (
        <div className="placeholder">
          <p>Upload a GraphML file to visualize the knowledge graph</p>
          <p>Supported features:</p>
          <ul>
            <li>Interactive radial, force-directed, and matrix views</li>
            <li>Click nodes to explore connections</li>
            <li>Toggle node/edge labels</li>
            <li>Detailed tooltips with metadata</li>
          </ul>
        </div>
      )}
    </div>
  );
};

export default GraphMLVisualizer;

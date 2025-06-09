import React, { useState, useEffect, useRef } from 'react';

const ChartsPanel = ({ filters }) => {
  const [timeSeriesData, setTimeSeriesData] = useState([]);
  const [distributionData, setDistributionData] = useState([]);
  const [topLossesData, setTopLossesData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedDimension, setSelectedDimension] = useState('business_line');
  const [chartType, setChartType] = useState('bar');

  const timeSeriesChartRef = useRef(null);
  const distributionChartRef = useRef(null);
  const topLossesChartRef = useRef(null);

  useEffect(() => {
    fetchChartData();
  }, [filters, selectedDimension]);

  const fetchChartData = async () => {
    setLoading(true);
    
    try {
      const queryParams = new URLSearchParams();
      Object.entries(filters).forEach(([key, value]) => {
        if (value !== null && value !== '' && value !== undefined) {
          if (Array.isArray(value) && value.length > 0) {
            value.forEach(v => queryParams.append(key, v));
          } else if (!Array.isArray(value)) {
            queryParams.append(key, value);
          }
        }
      });

      // Fetch time series data
      const timeSeriesResponse = await fetch(`http://localhost:8000/api/charts/time-series?${queryParams}`);
      const timeSeriesResult = await timeSeriesResponse.json();
      setTimeSeriesData(timeSeriesResult);

      // Fetch distribution data
      const distributionResponse = await fetch(`http://localhost:8000/api/charts/distribution?dimension=${selectedDimension}&${queryParams}`);
      const distributionResult = await distributionResponse.json();
      setDistributionData(distributionResult);

      // Fetch top losses data
      const topLossesResponse = await fetch(`http://localhost:8000/api/charts/top-losses?limit=10&${queryParams}`);
      const topLossesResult = await topLossesResponse.json();
      setTopLossesData(topLossesResult);

      // Initialize charts after data is loaded
      setTimeout(() => {
        initTimeSeriesChart();
        initDistributionChart();
        initTopLossesChart();
      }, 100);

    } catch (error) {
      console.error('Error fetching chart data:', error);
    } finally {
      setLoading(false);
    }
  };

  const initTimeSeriesChart = () => {
    if (!timeSeriesChartRef.current || timeSeriesData.length === 0) return;

    const canvas = timeSeriesChartRef.current;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Set canvas size
    canvas.width = canvas.offsetWidth * 2;
    canvas.height = canvas.offsetHeight * 2;
    ctx.scale(2, 2);
    
    // Chart dimensions
    const padding = { top: 20, right: 20, bottom: 40, left: 60 };
    const chartWidth = canvas.offsetWidth - padding.left - padding.right;
    const chartHeight = canvas.offsetHeight - padding.top - padding.bottom;
    
    // Data processing
    const maxLoss = Math.max(...timeSeriesData.map(d => d.total_loss));
    const maxEvents = Math.max(...timeSeriesData.map(d => d.event_count));
    
    // Draw background
    ctx.fillStyle = '#f9fafb';
    ctx.fillRect(0, 0, canvas.offsetWidth, canvas.offsetHeight);
    
    // Draw grid lines
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let i = 1; i <= 5; i++) {
      const y = padding.top + (chartHeight / 5) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartWidth, y);
      ctx.stroke();
    }
    
    // Draw loss amount line
    if (timeSeriesData.length > 1) {
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 3;
      ctx.beginPath();
      
      timeSeriesData.forEach((point, index) => {
        const x = padding.left + (chartWidth / (timeSeriesData.length - 1)) * index;
        const y = padding.top + chartHeight - (point.total_loss / maxLoss) * chartHeight;
        
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
      
      // Draw points
      ctx.fillStyle = '#3b82f6';
      timeSeriesData.forEach((point, index) => {
        const x = padding.left + (chartWidth / (timeSeriesData.length - 1)) * index;
        const y = padding.top + chartHeight - (point.total_loss / maxLoss) * chartHeight;
        
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fill();
      });
    }
    
    // Draw axes
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + chartHeight);
    ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
    ctx.stroke();
    
    // Draw labels
    ctx.fillStyle = '#374151';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    
    timeSeriesData.forEach((point, index) => {
      const x = padding.left + (chartWidth / (timeSeriesData.length - 1)) * index;
      const y = padding.top + chartHeight + 20;
      ctx.fillText(point.month, x, y);
    });
  };

  const initDistributionChart = () => {
    if (!distributionChartRef.current || distributionData.length === 0) return;

    const canvas = distributionChartRef.current;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Set canvas size
    canvas.width = canvas.offsetWidth * 2;
    canvas.height = canvas.offsetHeight * 2;
    ctx.scale(2, 2);
    
    // Chart dimensions
    const padding = { top: 20, right: 20, bottom: 80, left: 60 };
    const chartWidth = canvas.offsetWidth - padding.left - padding.right;
    const chartHeight = canvas.offsetHeight - padding.top - padding.bottom;
    
    const maxValue = Math.max(...distributionData.map(d => d.total_loss));
    const barWidth = chartWidth / distributionData.length * 0.8;
    const barSpacing = chartWidth / distributionData.length * 0.2;
    
    // Draw background
    ctx.fillStyle = '#f9fafb';
    ctx.fillRect(0, 0, canvas.offsetWidth, canvas.offsetHeight);
    
    // Draw bars
    distributionData.forEach((item, index) => {
      const barHeight = (item.total_loss / maxValue) * chartHeight;
      const x = padding.left + index * (barWidth + barSpacing) + barSpacing / 2;
      const y = padding.top + chartHeight - barHeight;
      
      // Draw bar
      const gradient = ctx.createLinearGradient(x, y, x, y + barHeight);
      gradient.addColorStop(0, '#3b82f6');
      gradient.addColorStop(1, '#1e40af');
      
      ctx.fillStyle = gradient;
      ctx.fillRect(x, y, barWidth, barHeight);
      
      // Draw value on top
      ctx.fillStyle = '#374151';
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(`$${(item.total_loss).toFixed(1)}M`, x + barWidth / 2, y - 5);
      
      // Draw category label
      ctx.save();
      ctx.translate(x + barWidth / 2, padding.top + chartHeight + 15);
      ctx.rotate(-Math.PI / 4);
      ctx.textAlign = 'right';
      ctx.fillText(item.category.substring(0, 15), 0, 0);
      ctx.restore();
    });
    
    // Draw axes
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + chartHeight);
    ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
    ctx.stroke();
  };

  const initTopLossesChart = () => {
    if (!topLossesChartRef.current || topLossesData.length === 0) return;

    const canvas = topLossesChartRef.current;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Set canvas size
    canvas.width = canvas.offsetWidth * 2;
    canvas.height = canvas.offsetHeight * 2;
    ctx.scale(2, 2);
    
    // Chart dimensions
    const padding = { top: 20, right: 100, bottom: 20, left: 150 };
    const chartWidth = canvas.offsetWidth - padding.left - padding.right;
    const chartHeight = canvas.offsetHeight - padding.top - padding.bottom;
    
    const maxValue = Math.max(...topLossesData.map(d => d.loss_amount___m_));
    const barHeight = chartHeight / topLossesData.length * 0.8;
    const barSpacing = chartHeight / topLossesData.length * 0.2;
    
    // Draw background
    ctx.fillStyle = '#f9fafb';
    ctx.fillRect(0, 0, canvas.offsetWidth, canvas.offsetHeight);
    
    // Draw bars
    topLossesData.forEach((item, index) => {
      const barWidth = (item.loss_amount___m_ / maxValue) * chartWidth;
      const x = padding.left;
      const y = padding.top + index * (barHeight + barSpacing) + barSpacing / 2;
      
      // Draw bar
      const color = index < 3 ? '#ef4444' : index < 6 ? '#f59e0b' : '#3b82f6';
      ctx.fillStyle = color;
      ctx.fillRect(x, y, barWidth, barHeight);
      
      // Draw company name
      ctx.fillStyle = '#374151';
      ctx.font = '11px Arial';
      ctx.textAlign = 'right';
      ctx.fillText(item.parent_name.substring(0, 20), x - 10, y + barHeight / 2 + 4);
      
      // Draw value
      ctx.textAlign = 'left';
      ctx.fillText(`$${item.loss_amount___m_.toFixed(1)}M`, x + barWidth + 10, y + barHeight / 2 + 4);
    });
  };

  const dimensions = [
    { value: 'business_line', label: 'Business Line' },
    { value: 'region', label: 'Region' },
    { value: 'risk_category', label: 'Risk Category' },
    { value: 'nfr_taxonomy', label: 'NFR Taxonomy' },
    { value: 'country', label: 'Country' }
  ];

  if (loading) {
    return (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {[...Array(3)].map((_, index) => (
          <div key={index} className="bg-white rounded-lg shadow p-6 animate-pulse">
            <div className="h-4 bg-gray-300 rounded w-32 mb-4"></div>
            <div className="h-64 bg-gray-300 rounded"></div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Chart Controls */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-medium text-gray-900">Analytics Dashboard</h3>
          <div className="flex items-center space-x-4">
            <select
              value={selectedDimension}
              onChange={(e) => setSelectedDimension(e.target.value)}
              className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {dimensions.map(dim => (
                <option key={dim.value} value={dim.value}>{dim.label}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Time Series Chart */}
        <div className="bg-white rounded-lg shadow p-6">
          <h4 className="text-md font-medium text-gray-900 mb-4">Loss Trends Over Time</h4>
          <div className="relative h-64">
            <canvas
              ref={timeSeriesChartRef}
              className="w-full h-full"
              style={{ maxWidth: '100%', maxHeight: '100%' }}
            />
          </div>
          <div className="mt-2 text-sm text-gray-600">
            Shows total losses by month over the selected time period
          </div>
        </div>

        {/* Distribution Chart */}
        <div className="bg-white rounded-lg shadow p-6">
          <h4 className="text-md font-medium text-gray-900 mb-4">
            Distribution by {dimensions.find(d => d.value === selectedDimension)?.label}
          </h4>
          <div className="relative h-64">
            <canvas
              ref={distributionChartRef}
              className="w-full h-full"
              style={{ maxWidth: '100%', maxHeight: '100%' }}
            />
          </div>
          <div className="mt-2 text-sm text-gray-600">
            Total losses aggregated by {selectedDimension.replace('_', ' ')}
          </div>
        </div>

        {/* Top Losses Chart */}
        <div className="lg:col-span-2 bg-white rounded-lg shadow p-6">
          <h4 className="text-md font-medium text-gray-900 mb-4">Top 10 Loss Events</h4>
          <div className="relative h-80">
            <canvas
              ref={topLossesChartRef}
              className="w-full h-full"
              style={{ maxWidth: '100%', maxHeight: '100%' }}
            />
          </div>
          <div className="mt-2 text-sm text-gray-600">
            Highest individual loss events ranked by loss amount
          </div>
        </div>
      </div>

      {/* Summary Statistics */}
      <div className="bg-white rounded-lg shadow p-6">
        <h4 className="text-md font-medium text-gray-900 mb-4">Quick Statistics</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {timeSeriesData.reduce((sum, item) => sum + item.event_count, 0)}
            </div>
            <div className="text-sm text-gray-600">Total Events</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-600">
              ${timeSeriesData.reduce((sum, item) => sum + item.total_loss, 0).toFixed(1)}M
            </div>
            <div className="text-sm text-gray-600">Total Losses</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {distributionData.length}
            </div>
            <div className="text-sm text-gray-600">Categories</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              ${topLossesData.length > 0 ? topLossesData[0].loss_amount___m_.toFixed(1) : 0}M
            </div>
            <div className="text-sm text-gray-600">Highest Loss</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChartsPanel;
import React, { useState, useEffect } from 'react';

const SummaryCards = ({ filters }) => {
  const [summaryData, setSummaryData] = useState({
    total_losses: 0,
    total_events: 0,
    average_loss: 0,
    affected_firms: 0,
    highest_loss: 0,
    recent_events: 0
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchSummaryData();
  }, [filters]);

  const fetchSummaryData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const queryParams = new URLSearchParams();
      
      // Add filters to query params
      Object.entries(filters).forEach(([key, value]) => {
        if (value !== null && value !== '' && value !== undefined) {
          if (Array.isArray(value) && value.length > 0) {
            value.forEach(v => queryParams.append(key, v));
          } else if (!Array.isArray(value)) {
            queryParams.append(key, value);
          }
        }
      });

      const response = await fetch(`http://localhost:8000/api/summary?${queryParams}`);
      if (!response.ok) {
        throw new Error('Failed to fetch summary data');
      }
      
      const data = await response.json();
      setSummaryData(data);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching summary data:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(amount * 1000000); // Convert millions to actual value
  };

  const formatNumber = (num) => {
    return new Intl.NumberFormat('en-US').format(num);
  };

  const cards = [
    {
      title: 'Total Losses',
      value: formatCurrency(summaryData.total_losses),
      icon: '💰',
      color: 'bg-red-500',
      change: null,
      description: 'Total financial losses'
    },
    {
      title: 'Total Events',
      value: formatNumber(summaryData.total_events),
      icon: '📊',
      color: 'bg-blue-500',
      change: null,
      description: 'Number of loss events'
    },
    {
      title: 'Average Loss',
      value: formatCurrency(summaryData.average_loss),
      icon: '📈',
      color: 'bg-yellow-500',
      change: null,
      description: 'Average loss per event'
    },
    {
      title: 'Affected Firms',
      value: formatNumber(summaryData.affected_firms),
      icon: '🏢',
      color: 'bg-purple-500',
      change: null,
      description: 'Number of affected firms'
    },
    {
      title: 'Highest Loss',
      value: formatCurrency(summaryData.highest_loss),
      icon: '⚠️',
      color: 'bg-orange-500',
      change: null,
      description: 'Single highest loss event'
    },
    {
      title: 'Recent Events',
      value: formatNumber(summaryData.recent_events),
      icon: '🕒',
      color: 'bg-green-500',
      change: null,
      description: 'Events in last 30 days'
    }
  ];

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[...Array(6)].map((_, index) => (
          <div key={index} className="bg-white rounded-lg shadow p-6 animate-pulse">
            <div className="flex items-center">
              <div className="w-8 h-8 bg-gray-300 rounded-full"></div>
              <div className="ml-4 flex-1">
                <div className="h-4 bg-gray-300 rounded w-24 mb-2"></div>
                <div className="h-6 bg-gray-300 rounded w-32"></div>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex">
          <div className="text-red-400 text-xl mr-3">⚠️</div>
          <div>
            <h3 className="text-red-800 font-medium">Error loading summary data</h3>
            <p className="text-red-600 text-sm mt-1">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {cards.map((card, index) => (
        <div
          key={index}
          className="bg-white rounded-lg shadow hover:shadow-lg transition-shadow duration-200 p-6 border border-gray-200"
        >
          <div className="flex items-center">
            <div className={`flex-shrink-0 ${card.color} rounded-lg p-3`}>
              <div className="text-white text-xl">{card.icon}</div>
            </div>
            <div className="ml-4 flex-1">
              <div className="text-sm font-medium text-gray-600 uppercase tracking-wide">
                {card.title}
              </div>
              <div className="text-2xl font-bold text-gray-900 mt-1">
                {card.value}
              </div>
              {card.change && (
                <div className={`text-sm mt-1 ${
                  card.change.startsWith('+') ? 'text-green-600' : 'text-red-600'
                }`}>
                  {card.change}
                </div>
              )}
            </div>
          </div>
          <div className="mt-4 text-sm text-gray-500">
            {card.description}
          </div>
        </div>
      ))}
    </div>
  );
};

export default SummaryCards;
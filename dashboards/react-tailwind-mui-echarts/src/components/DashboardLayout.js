import React, { useState, useEffect } from 'react';
import SummaryCards from './SummaryCards';
import FiltersPanel from './FiltersPanel';
import DataTable from './DataTable';
import ChartsPanel from './ChartsPanel';
import SearchBar from './SearchBar';

const DashboardLayout = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [filters, setFilters] = useState({
    search: '',
    loss_amount_min: null,
    loss_amount_max: null,
    date_from: null,
    date_to: null,
    business_lines: [],
    regions: [],
    countries: [],
    risk_categories: [],
    nfr_taxonomies: [],
    ubs_divisions: []
  });
  const [showFilters, setShowFilters] = useState(false);

  const tabs = [
    { id: 'overview', name: 'Overview', icon: '📊' },
    { id: 'data', name: 'Data Explorer', icon: '📋' },
    { id: 'analytics', name: 'Analytics', icon: '🔍' },
    { id: 'reports', name: 'Reports', icon: '📄' }
  ];

  const handleFilterChange = (newFilters) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
  };

  const clearFilters = () => {
    setFilters({
      search: '',
      loss_amount_min: null,
      loss_amount_max: null,
      date_from: null,
      date_to: null,
      business_lines: [],
      regions: [],
      countries: [],
      risk_categories: [],
      nfr_taxonomies: [],
      ubs_divisions: []
    });
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">
                External Loss Data Dashboard
              </h1>
            </div>
            
            <div className="flex items-center space-x-4">
              <SearchBar 
                value={filters.search}
                onChange={(search) => handleFilterChange({ search })}
              />
              
              <button
                onClick={() => setShowFilters(!showFilters)}
                className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.207A1 1 0 013 6.5V4z" />
                </svg>
                Filters
              </button>
              
              <button className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Export
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex gap-6">
          {/* Sidebar Filters */}
          {showFilters && (
            <div className="w-80 flex-shrink-0">
              <FiltersPanel 
                filters={filters}
                onFilterChange={handleFilterChange}
                onClearFilters={clearFilters}
              />
            </div>
          )}

          {/* Main Content */}
          <div className="flex-1">
            {/* Tab Navigation */}
            <div className="border-b border-gray-200 mb-6">
              <nav className="-mb-px flex space-x-8">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`group inline-flex items-center py-2 px-1 border-b-2 font-medium text-sm ${
                      activeTab === tab.id
                        ? 'border-blue-500 text-blue-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }`}
                  >
                    <span className="mr-2 text-lg">{tab.icon}</span>
                    {tab.name}
                  </button>
                ))}
              </nav>
            </div>

            {/* Tab Content */}
            <div className="space-y-6">
              {activeTab === 'overview' && (
                <>
                  <SummaryCards filters={filters} />
                  <ChartsPanel filters={filters} />
                </>
              )}
              
              {activeTab === 'data' && (
                <DataTable filters={filters} />
              )}
              
              {activeTab === 'analytics' && (
                <div className="bg-white rounded-lg shadow p-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Advanced Analytics</h3>
                  <div className="text-gray-500 text-center py-12">
                    <div className="text-6xl mb-4">🔍</div>
                    <p>Advanced analytics features coming soon...</p>
                    <p className="text-sm mt-2">ML insights, clustering, and statistical analysis</p>
                  </div>
                </div>
              )}
              
              {activeTab === 'reports' && (
                <div className="bg-white rounded-lg shadow p-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Reports</h3>
                  <div className="text-gray-500 text-center py-12">
                    <div className="text-6xl mb-4">📄</div>
                    <p>Custom report builder coming soon...</p>
                    <p className="text-sm mt-2">Generate PDF reports with charts and analysis</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardLayout;
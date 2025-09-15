import React from 'react';
import { motion } from 'framer-motion';
import { useDashboard } from '../contexts/DashboardContext';
import StatsCards from './StatsCards';
import DataTable from './DataTable';
import SearchBar from './SearchBar';
import CombinedDetailView from './CombinedDetailView';

const Dashboard = ({ user = "default_user" }) => {
  const { state, actions } = useDashboard();

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="w-full h-screen flex flex-col p-4">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6"
        >
          <p className="text-gray-600">
            Welcome back, {user}. Monitor and analyze risk data across all datasets.
          </p>
        </motion.div>

        {/* Search Bar - Full Width */}
        <motion.div 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-6"
        >
          <SearchBar />
        </motion.div>

        {/* Stats Cards */}
        <motion.div 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-6"
        >
          <StatsCards />
        </motion.div>

        {/* Data Table */}
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="card p-0"
        >
          <div className="p-4 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold text-gray-900">
                  {state.selectedDataset.replace('_', ' ').toUpperCase()} Dataset
                </h2>
                <p className="text-sm text-gray-600 mt-1">
                  {state.totalRecords} records found
                  {state.searchQuery && ` matching "${state.searchQuery}"`}
                </p>
              </div>
              
              {/* Quick Filters */}
              <div className="flex items-center space-x-4">
                {state.taxonomies.ai_taxonomies && state.taxonomies.ai_taxonomies.length > 0 && (
                  <select
                    value={state.selectedAiTaxonomy}
                    onChange={(e) => actions.setAiTaxonomy(e.target.value)}
                    className="input-field text-sm"
                  >
                    <option value="">All AI Types</option>
                    {state.taxonomies.ai_taxonomies.map(taxonomy => (
                      <option key={taxonomy} value={taxonomy}>{taxonomy}</option>
                    ))}
                  </select>
                )}
                
                {state.taxonomies.erms_taxonomies && state.taxonomies.erms_taxonomies.length > 0 && (
                  <select
                    value={state.selectedErmsTaxonomy}
                    onChange={(e) => actions.setErmsTaxonomy(e.target.value)}
                    className="input-field text-sm"
                  >
                    <option value="">All ERMS Types</option>
                    {state.taxonomies.erms_taxonomies.map(taxonomy => (
                      <option key={taxonomy} value={taxonomy}>{taxonomy}</option>
                    ))}
                  </select>
                )}
              </div>
            </div>
          </div>
          
          <DataTable />
        </motion.div>

        {/* Error Display */}
        {state.error && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="fixed bottom-4 right-4 bg-primary-600 text-white px-6 py-3 shadow-lg z-50"
          >
            <div className="flex items-center">
              <span className="mr-2">⚠️</span>
              {state.error}
              <button
                onClick={actions.clearError}
                className="ml-4 text-white hover:text-gray-200"
              >
                ✕
              </button>
            </div>
          </motion.div>
        )}
      </div>

      {/* Combined Detail View Modal */}
      <CombinedDetailView />
    </div>
  );
};

export default Dashboard;
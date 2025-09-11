import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { BarChart, AlertTriangle, Shield, FileText } from 'lucide-react';
import { useDashboard } from '../contexts/DashboardContext';

const datasetConfig = {
  external_loss: {
    name: 'External Loss',
    icon: AlertTriangle,
    color: 'text-red-600',
    bgColor: 'bg-red-50',
    borderColor: 'border-red-200'
  },
  internal_loss: {
    name: 'Internal Loss',
    icon: BarChart,
    color: 'text-orange-600',
    bgColor: 'bg-orange-50',
    borderColor: 'border-orange-200'
  },
  issues: {
    name: 'Issues',
    icon: FileText,
    color: 'text-yellow-600',
    bgColor: 'bg-yellow-50',
    borderColor: 'border-yellow-200'
  },
  controls: {
    name: 'Controls',
    icon: Shield,
    color: 'text-green-600',
    bgColor: 'bg-green-50',
    borderColor: 'border-green-200'
  }
};

const StatCard = ({ dataset, stats, isSelected, onClick }) => {
  const config = datasetConfig[dataset];
  const Icon = config.icon;

  return (
    <motion.div
      whileHover={{ scale: 1.02, y: -2 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className={`
        card card-hover cursor-pointer p-3 transition-all duration-200
        ${isSelected 
          ? `${config.bgColor} ${config.borderColor} border-2` 
          : 'bg-white hover:bg-gray-50'
        }
      `}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <div className={`p-1.5 ${config.bgColor}`}>
            <Icon className={`h-4 w-4 ${config.color}`} />
          </div>
          <h3 className="text-sm font-semibold text-gray-900">
            {config.name}
          </h3>
        </div>
        {isSelected && (
          <div className={`px-1.5 py-0.5 text-xs font-medium ${config.color} ${config.bgColor}`}>
            Active
          </div>
        )}
      </div>
      
      {stats ? (
        <div className="space-y-1.5">
          <div className="flex items-baseline justify-between">
            <span className="text-xl font-bold text-gray-900">
              {stats.total_records?.toLocaleString() || 0}
            </span>
            <span className="text-xs text-gray-500">records</span>
          </div>
        </div>
      ) : (
        <div className="space-y-1.5">
          <div className="h-7 bg-gray-200 animate-pulse"></div>
          <div className="flex items-center justify-between pt-1.5">
            <div className="h-4 w-12 bg-gray-200 animate-pulse"></div>
            <div className="h-4 w-12 bg-gray-200 animate-pulse"></div>
          </div>
        </div>
      )}
    </motion.div>
  );
};

const StatsCards = () => {
  const { state, actions } = useDashboard();
  const datasets = ['external_loss', 'internal_loss', 'issues', 'controls'];

  useEffect(() => {
    // Fetch stats for all datasets on mount
    datasets.forEach(dataset => {
      if (!state.stats[dataset]) {
        actions.fetchStats(dataset);
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {datasets.map((dataset, index) => (
        <motion.div
          key={dataset}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
        >
          <StatCard
            dataset={dataset}
            stats={state.stats[dataset]}
            isSelected={state.selectedDataset === dataset}
            onClick={() => actions.setSelectedDataset(dataset)}
          />
        </motion.div>
      ))}
    </div>
  );
};

export default StatsCards;
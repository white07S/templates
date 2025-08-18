import React, { useState, useEffect } from 'react';
import { Database, Zap, TrendingUp, FileText } from 'lucide-react';
import { motion } from 'framer-motion';
import { taskAPI } from '../../../services/api';
import LoadingSpinner from '../../../components/common/LoadingSpinner';
import ErrorMessage from '../../../components/common/ErrorMessage';

const PlatformOfferings = () => {
  const [offerings, setOfferings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchOfferings();
  }, []);

  const fetchOfferings = async () => {
    try {
      setLoading(true);
      setError('');
      const response = await taskAPI.getAvailableTasks();
      const transformedData = response.data.data_types.map(dataType => ({
        name: dataType.name,
        display_name: dataType.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        description: `Advanced ${dataType.name.replace(/_/g, ' ')} processing capabilities`,
        tasks: dataType.tasks.map(taskName => ({
          name: taskName,
          display_name: taskName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
        }))
      }));
      setOfferings(transformedData);
    } catch (error) {
      console.error('Error fetching platform offerings:', error);
      setError('Failed to load platform offerings. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const getDataTypeIcon = (dataType) => {
    switch (dataType.toLowerCase()) {
      case 'controls':
        return <Database size={24} />;
      case 'risks':
        return <TrendingUp size={24} />;
      case 'processes':
        return <Zap size={24} />;
      default:
        return <FileText size={24} />;
    }
  };


  if (loading) {
    return <LoadingSpinner size="large" message="Loading platform offerings..." />;
  }

  if (error) {
    return <ErrorMessage message={error} onRetry={fetchOfferings} />;
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-black mb-2">Platform Offerings</h2>
        <p className="text-gray-600">
          Explore our comprehensive suite of data processing and analytics capabilities
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {offerings.map((dataType, index) => (
          <motion.div
            key={dataType.name}
            className="border-2 border-gray-200 p-6 hover:border-red-600 transition-colors"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
          >
            <div className="flex items-start space-x-4">
              <div className="text-red-600">
                {getDataTypeIcon(dataType.name)}
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-bold text-black mb-1">{dataType.display_name}</h3>
                <p className="text-gray-600 text-sm mb-3">{dataType.description}</p>
              </div>
            </div>

            <div className="mt-4">
              <h4 className="font-medium text-black mb-2">Available Tasks</h4>
              <div className="space-y-1">
                {dataType.tasks.map((task) => (
                  <div key={task.name} className="flex items-center text-sm">
                    <div className="w-2 h-2 bg-red-600 mr-2"></div>
                    <span className="text-gray-700">{task.display_name}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-gray-200">
              <span className="text-sm text-gray-600">
                {dataType.tasks.length} {dataType.tasks.length === 1 ? 'task' : 'tasks'} available
              </span>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

export default PlatformOfferings;
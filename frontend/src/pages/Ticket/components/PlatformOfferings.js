import React, { useState, useEffect } from 'react';
import { Database, Zap, TrendingUp, FileText } from 'lucide-react';
import { motion } from 'framer-motion';
import { taskAPI } from '../../../services/api';
import LoadingSpinner from '../../../components/common/LoadingSpinner';
import ErrorMessage from '../../../components/common/ErrorMessage';
import './PlatformOfferings.css';

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

  const getTaskDescription = (taskName) => {
    const descriptions = {
      taxonomy_mapping: 'Map data to standardized taxonomy structures',
      ai_insights: 'Generate AI-powered insights and recommendations',
      risk_assessment: 'Analyze and assess potential risk factors',
      compliance_check: 'Verify compliance with regulatory requirements',
      process_optimization: 'Identify opportunities for process improvement',
      data_validation: 'Validate data quality and consistency'
    };
    return descriptions[taskName] || 'Process data with advanced analytics';
  };

  if (loading) {
    return <LoadingSpinner size="large" message="Loading platform offerings..." />;
  }

  if (error) {
    return <ErrorMessage message={error} onRetry={fetchOfferings} />;
  }

  return (
    <div className="platform-offerings">
      <div className="offerings-header">
        <h2 className="text-heading">Platform Offerings</h2>
        <p className="text-body">
          Explore our comprehensive suite of data processing and analytics capabilities
        </p>
      </div>

      <div className="offerings-grid">
        {offerings.map((dataType, index) => (
          <motion.div
            key={dataType.name}
            className="offering-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
          >
            <div className="offering-header">
              <div className="offering-icon">
                {getDataTypeIcon(dataType.name)}
              </div>
              <div className="offering-title">
                <h3 className="text-subheading">{dataType.display_name}</h3>
                <p className="text-caption">{dataType.description}</p>
              </div>
            </div>

            <div className="offering-tasks">
              <h4 className="tasks-title">Available Tasks</h4>
              <div className="tasks-list">
                {dataType.tasks.map((task) => (
                  <div key={task.name} className="task-item">
                    <div className="task-name">{task.display_name}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="offering-footer">
              <span className="task-count">
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
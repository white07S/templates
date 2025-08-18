import React, { useState, useEffect } from 'react';
import { RefreshCw, Download, Clock, CheckCircle, AlertCircle } from 'lucide-react';
import { motion } from 'framer-motion';
import { taskAPI } from '../../../services/api';
import { formatDate, downloadFile, formatDataType } from '../../../utils/helpers';
import Button from '../../../components/common/Button';
import LoadingSpinner from '../../../components/common/LoadingSpinner';
import ErrorMessage from '../../../components/common/ErrorMessage';

const TaskStatus = ({ user, refreshTrigger }) => {
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');
  const [downloadingTask, setDownloadingTask] = useState(null);

  useEffect(() => {
    fetchTasks();
  }, [refreshTrigger]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const interval = setInterval(fetchTasks, 30000); // Auto-refresh every 30 seconds
    return () => clearInterval(interval);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const fetchTasks = async (showRefreshIndicator = false) => {
    if (showRefreshIndicator) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }

    try {
      // Fetch all user tasks from backend
      const response = await taskAPI.getUserTasks(user);
      const userTasks = response.data.tasks || [];

      // Transform the data to match our component's expected format
      const transformedTasks = userTasks.map(task => ({
        task_id: task.task_id,
        data_type: task.data_type,
        tasks: task.tasks,
        status: task.status,
        submitted_at: task.submitted_at,
        file_name: task.file_name
      }));

      setTasks(transformedTasks);
      setError('');
    } catch (error) {
      const errorMessage = error.response?.data?.detail || 'Failed to fetch tasks';
      setError(errorMessage);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleRefresh = () => {
    fetchTasks(true);
  };

  const handleDownload = async (taskId, fileName) => {
    setDownloadingTask(taskId);
    try {
      const response = await taskAPI.getTaskResult(taskId);
      downloadFile(response.data, fileName || `result_${taskId}.xlsx`);
    } catch (error) {
      setError('Failed to download result file');
    } finally {
      setDownloadingTask(null);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle size={20} className="text-green-600" />;
      case 'pending':
        return <Clock size={20} className="text-yellow-600" />;
      case 'error':
        return <AlertCircle size={20} className="text-red-600" />;
      default:
        return <Clock size={20} className="text-gray-600" />;
    }
  };

  const getStatusClass = (status) => {
    switch (status) {
      case 'completed':
        return 'bg-green-50 text-green-700 border-green-200';
      case 'pending':
        return 'bg-yellow-50 text-yellow-700 border-yellow-200';
      case 'error':
        return 'bg-red-50 text-red-700 border-red-200';
      default:
        return 'bg-gray-50 text-gray-700 border-gray-200';
    }
  };

  if (loading && tasks.length === 0) {
    return <LoadingSpinner size="large" message="Loading your tasks..." />;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-black">My Tasks</h2>
          <p className="text-gray-600">Track the status of your submitted tasks</p>
        </div>
        <Button
          variant="ghost"
          size="medium"
          onClick={handleRefresh}
          loading={refreshing}
          icon={<RefreshCw size={18} />}
        >
          Refresh
        </Button>
      </div>

      {error && <ErrorMessage message={error} onClose={() => setError('')} />}

      {tasks.length === 0 ? (
        <div className="text-center py-12 bg-gray-50 border-2 border-gray-200">
          <p className="text-gray-600">No tasks found. Submit your first task to get started!</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {tasks.map((task, index) => (
            <motion.div
              key={task.task_id}
              className="bg-white border-2 border-gray-200 p-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
            >
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="font-bold text-black">Task ID: {task.task_id.substring(0, 8)}...</h3>
                  <p className="text-sm text-gray-600">{formatDate(task.submitted_at)}</p>
                </div>
                <div className={`flex items-center space-x-2 px-3 py-1 border ${getStatusClass(task.status)}`}>
                  {getStatusIcon(task.status)}
                  <span className="font-medium capitalize">{task.status}</span>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex">
                  <span className="text-gray-600 w-24">Data Type:</span>
                  <span className="text-black font-medium">{formatDataType(task.data_type)}</span>
                </div>
                <div className="flex">
                  <span className="text-gray-600 w-24">Tasks:</span>
                  <span className="text-black">{task.tasks.join(', ')}</span>
                </div>
                <div className="flex">
                  <span className="text-gray-600 w-24">File:</span>
                  <span className="text-black">{task.file_name}</span>
                </div>
              </div>

              {task.status === 'completed' && (
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <Button
                    variant="primary"
                    size="small"
                    icon={<Download size={16} />}
                    onClick={() => handleDownload(task.task_id, `result_${task.file_name}`)}
                    loading={downloadingTask === task.task_id}
                  >
                    Download Result
                  </Button>
                </div>
              )}
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
};

export default TaskStatus;
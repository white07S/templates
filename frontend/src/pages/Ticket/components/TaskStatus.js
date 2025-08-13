import React, { useState, useEffect } from 'react';
import { RefreshCw, Download, Clock, CheckCircle, AlertCircle } from 'lucide-react';
import { motion } from 'framer-motion';
import { taskAPI } from '../../../services/api';
import { formatDate, downloadFile, formatDataType } from '../../../utils/helpers';
import Button from '../../../components/common/Button';
import LoadingSpinner from '../../../components/common/LoadingSpinner';
import ErrorMessage from '../../../components/common/ErrorMessage';
import './TaskStatus.css';

const TaskStatus = ({ user, refreshTrigger }) => {
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');
  const [downloadingTask, setDownloadingTask] = useState(null);

  useEffect(() => {
    fetchTasks();
  }, [refreshTrigger]);

  useEffect(() => {
    const interval = setInterval(fetchTasks, 30000); // Auto-refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

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
        return <CheckCircle size={20} className="status-icon-completed" />;
      case 'pending':
        return <Clock size={20} className="status-icon-pending" />;
      case 'error':
        return <AlertCircle size={20} className="status-icon-error" />;
      default:
        return <Clock size={20} className="status-icon-pending" />;
    }
  };

  const getStatusClass = (status) => {
    switch (status) {
      case 'completed':
        return 'status-badge-completed';
      case 'pending':
        return 'status-badge-pending';
      case 'error':
        return 'status-badge-error';
      default:
        return 'status-badge-pending';
    }
  };

  if (loading && tasks.length === 0) {
    return <LoadingSpinner size="large" message="Loading your tasks..." />;
  }

  return (
    <div className="task-status">
      <div className="status-header">
        <div>
          <h2 className="text-heading">My Tasks</h2>
          <p className="text-body">Track the status of your submitted tasks</p>
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
        <div className="no-tasks">
          <p className="text-body">No tasks found. Submit your first task to get started!</p>
        </div>
      ) : (
        <div className="tasks-grid">
          {tasks.map((task, index) => (
            <motion.div
              key={task.task_id}
              className="task-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
            >
              <div className="task-card-header">
                <div className="task-info">
                  <h3 className="task-id">Task ID: {task.task_id.substring(0, 8)}...</h3>
                  <p className="task-date">{formatDate(task.submitted_at)}</p>
                </div>
                <div className={`status-badge ${getStatusClass(task.status)}`}>
                  {getStatusIcon(task.status)}
                  <span>{task.status}</span>
                </div>
              </div>

              <div className="task-details">
                <div className="detail-row">
                  <span className="detail-label">Data Type:</span>
                  <span className="detail-value">{formatDataType(task.data_type)}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Tasks:</span>
                  <span className="detail-value">{task.tasks.join(', ')}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">File:</span>
                  <span className="detail-value">{task.file_name}</span>
                </div>
              </div>

              {task.status === 'completed' && (
                <div className="task-actions">
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
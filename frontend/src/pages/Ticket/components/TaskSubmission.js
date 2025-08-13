import React, { useState, useEffect } from 'react';
import { Send } from 'lucide-react';
import { motion } from 'framer-motion';
import { taskAPI } from '../../../services/api';
import DataTypeSelector from './DataTypeSelector';
import TaskSelector from './TaskSelector';
import FileUpload from './FileUpload';
import Button from '../../../components/common/Button';
import ErrorMessage from '../../../components/common/ErrorMessage';
import LoadingSpinner from '../../../components/common/LoadingSpinner';
import './TaskSubmission.css';

const TaskSubmission = ({ user, onTaskSubmitted }) => {
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  
  const [dataTypes, setDataTypes] = useState([]);
  const [formData, setFormData] = useState({
    dataType: '',
    selectedTasks: [],
    file: null
  });

  useEffect(() => {
    fetchAvailableTasks();
  }, []);

  const fetchAvailableTasks = async () => {
    try {
      const response = await taskAPI.getAvailableTasks();
      setDataTypes(response.data.data_types);
    } catch (error) {
      setError('Failed to fetch available tasks');
    } finally {
      setLoading(false);
    }
  };

  const handleDataTypeChange = (dataType) => {
    setFormData({
      dataType,
      selectedTasks: [],
      file: null
    });
    setError('');
    setSuccess('');
  };

  const handleTasksChange = (tasks) => {
    setFormData(prev => ({ ...prev, selectedTasks: tasks }));
  };

  const handleFileSelect = (file) => {
    setFormData(prev => ({ ...prev, file }));
    setError('');
  };

  const handleFileRemove = () => {
    setFormData(prev => ({ ...prev, file: null }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    if (!formData.dataType || formData.selectedTasks.length === 0 || !formData.file) {
      setError('Please fill in all required fields');
      return;
    }

    setSubmitting(true);
    
    try {
      const submitData = {
        username: user,
        data_type: formData.dataType,
        tasks: formData.selectedTasks.join(','),
        file: formData.file
      };

      const response = await taskAPI.submitTask(submitData);
      
      setSuccess(`Task submitted successfully! Task ID: ${response.data.task_id}`);
      setFormData({
        dataType: '',
        selectedTasks: [],
        file: null
      });
      
      if (onTaskSubmitted) {
        setTimeout(() => {
          onTaskSubmitted();
        }, 2000);
      }
    } catch (error) {
      const errorDetail = error.response?.data?.detail;
      if (typeof errorDetail === 'string') {
        setError(errorDetail);
      } else if (Array.isArray(errorDetail)) {
        const errorMessages = errorDetail.map(err => err.msg || 'Validation error').join(', ');
        setError(`Validation errors: ${errorMessages}`);
      } else if (errorDetail && typeof errorDetail === 'object') {
        setError(errorDetail.msg || JSON.stringify(errorDetail));
      } else {
        setError('Failed to submit task');
      }
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return <LoadingSpinner size="large" message="Loading task options..." />;
  }

  const selectedDataType = dataTypes.find(dt => dt.name === formData.dataType);
  const availableTasks = selectedDataType?.tasks || [];

  return (
    <div className="task-submission">
      <div className="submission-header">
        <h2 className="text-heading">Submit New Task</h2>
        <p className="text-body">Upload your data file and select processing tasks</p>
      </div>

      {error && <ErrorMessage message={error} onClose={() => setError('')} />}
      {success && <ErrorMessage message={success} type="success" onClose={() => setSuccess('')} />}

      <form onSubmit={handleSubmit} className="submission-form">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <DataTypeSelector
            value={formData.dataType}
            onChange={handleDataTypeChange}
            dataTypes={dataTypes}
            disabled={submitting}
          />
        </motion.div>

        {formData.dataType && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <TaskSelector
              availableTasks={availableTasks}
              selectedTasks={formData.selectedTasks}
              onChange={handleTasksChange}
              disabled={submitting}
            />
          </motion.div>
        )}

        {formData.dataType && formData.selectedTasks.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.2 }}
          >
            <FileUpload
              onFileSelect={handleFileSelect}
              requiredColumns={[]}
              disabled={submitting}
              file={formData.file}
              onRemove={handleFileRemove}
            />
          </motion.div>
        )}

        {formData.dataType && formData.selectedTasks.length > 0 && formData.file && (
          <motion.div
            className="submission-actions"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.3 }}
          >
            <Button
              type="submit"
              variant="primary"
              size="large"
              loading={submitting}
              icon={<Send size={20} />}
            >
              Submit Task
            </Button>
          </motion.div>
        )}
      </form>
    </div>
  );
};

export default TaskSubmission;
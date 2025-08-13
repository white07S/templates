import React, { useState, useEffect } from 'react';
import { FileText, Clock, CheckCircle, TrendingUp } from 'lucide-react';
import { motion } from 'framer-motion';
import { taskAPI } from '../../../services/api';
import LoadingSpinner from '../../../components/common/LoadingSpinner';
import ErrorMessage from '../../../components/common/ErrorMessage';
import { formatDate, formatDataType } from '../../../utils/helpers';
import './TicketDashboard.css';

const TicketDashboard = ({ user, setActiveTab, refreshTrigger }) => {
  const [stats, setStats] = useState({
    totalTasks: 0,
    pendingTasks: 0,
    completedTasks: 0,
    availableOfferings: 0
  });
  const [recentTasks, setRecentTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchDashboardData();
  }, [refreshTrigger]);

  const fetchDashboardData = async () => {
    setLoading(true);
    setError('');
    
    try {
      // Fetch available offerings
      const offeringsResponse = await taskAPI.getAvailableTasks();
      const offerings = offeringsResponse.data.data_types.reduce((acc, dt) => acc + dt.tasks.length, 0);
      
      // Fetch user tasks to get real statistics
      if (user) {
        try {
          const userTasksResponse = await taskAPI.getUserTasks(user);
          const userTasks = userTasksResponse.data.tasks || [];
          
          // Calculate real statistics
          const totalTasks = userTasks.length;
          const pendingTasks = userTasks.filter(task => task.status === 'pending').length;
          const completedTasks = userTasks.filter(task => task.status === 'completed').length;
          
          // Get recent tasks (last 3)
          const recent = userTasks.slice(0, 3);
          
          setStats({
            totalTasks,
            pendingTasks,
            completedTasks,
            availableOfferings: offerings
          });
          setRecentTasks(recent);
        } catch (userTasksError) {
          console.error('Error fetching user tasks:', userTasksError);
          const errorMsg = userTasksError.response?.data?.detail || 'Failed to fetch user tasks';
          setError(errorMsg);
          
          // Fallback to default stats if user tasks fetch fails
          setStats({
            totalTasks: 0,
            pendingTasks: 0,
            completedTasks: 0,
            availableOfferings: offerings
          });
          setRecentTasks([]);
        }
      } else {
        setError('User authentication required');
        setStats({
          totalTasks: 0,
          pendingTasks: 0,
          completedTasks: 0,
          availableOfferings: offerings
        });
        setRecentTasks([]);
      }
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setError('Failed to load dashboard data');
      setStats({
        totalTasks: 0,
        pendingTasks: 0,
        completedTasks: 0,
        availableOfferings: 0
      });
      setRecentTasks([]);
    } finally {
      setLoading(false);
    }
  };

  const statCards = [
    {
      title: 'Total Tasks',
      value: stats.totalTasks,
      icon: <FileText size={24} />,
      color: 'var(--ubs-black)'
    },
    {
      title: 'Pending Tasks',
      value: stats.pendingTasks,
      icon: <Clock size={24} />,
      color: 'var(--ubs-gray-dark)'
    },
    {
      title: 'Completed Tasks',
      value: stats.completedTasks,
      icon: <CheckCircle size={24} />,
      color: 'var(--ubs-red)'
    },
    {
      title: 'Available Offerings',
      value: stats.availableOfferings,
      icon: <TrendingUp size={24} />,
      color: 'var(--ubs-black)'
    }
  ];

  if (loading) {
    return <LoadingSpinner size="large" message="Loading dashboard..." />;
  }

  return (
    <div className="ticket-dashboard">
      <div className="dashboard-welcome">
        <h2 className="text-heading">Welcome back, {user?.username}</h2>
        <p className="text-body">Here's an overview of your task processing activities</p>
      </div>

      {error && <ErrorMessage message={error} onClose={() => setError('')} />}

      <div className="dashboard-stats">
        {statCards.map((stat, index) => (
          <motion.div
            key={stat.title}
            className="stat-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
          >
            <div className="stat-icon" style={{ color: stat.color }}>
              {stat.icon}
            </div>
            <div className="stat-content">
              <h3 className="stat-value">{stat.value}</h3>
              <p className="stat-title">{stat.title}</p>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="dashboard-actions">
        <h3 className="text-subheading">Quick Actions</h3>
        <div className="action-cards">
          <motion.div
            className="action-card"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setActiveTab('submit')}
          >
            <h4>Submit New Task</h4>
            <p>Upload your data files and select processing tasks</p>
          </motion.div>
          <motion.div
            className="action-card"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setActiveTab('status')}
          >
            <h4>View My Tasks</h4>
            <p>Track the status of your submitted tasks</p>
          </motion.div>
          <motion.div
            className="action-card"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setActiveTab('offerings')}
          >
            <h4>Platform Offerings</h4>
            <p>Explore available data types and processing tasks</p>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default TicketDashboard;
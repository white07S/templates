import React, { useState, useEffect } from 'react';
import { FileText, Clock, CheckCircle, TrendingUp, ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';
import { taskAPI } from '../../../services/api';
import LoadingSpinner from '../../../components/common/LoadingSpinner';
import ErrorMessage from '../../../components/common/ErrorMessage';

const TicketDashboard = ({ user, setActiveTab, refreshTrigger }) => {
  const [stats, setStats] = useState({
    totalTasks: 0,
    pendingTasks: 0,
    completedTasks: 0,
    availableOfferings: 0
  });
  // const [recentTasks, setRecentTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchDashboardData();
  }, [refreshTrigger]); // eslint-disable-line react-hooks/exhaustive-deps

  const fetchDashboardData = async () => {
    setLoading(true);
    setError('');
    
    try {
      const offeringsResponse = await taskAPI.getAvailableTasks();
      const offerings = offeringsResponse.data.data_types.reduce((acc, dt) => acc + dt.tasks.length, 0);
      
      if (user) {
        try {
          const userTasksResponse = await taskAPI.getUserTasks(user);
          const userTasks = userTasksResponse.data.tasks || [];
          
          const totalTasks = userTasks.length;
          const pendingTasks = userTasks.filter(task => task.status === 'pending').length;
          const completedTasks = userTasks.filter(task => task.status === 'completed').length;
          
          // const recent = userTasks.slice(0, 3);
          
          setStats({
            totalTasks,
            pendingTasks,
            completedTasks,
            availableOfferings: offerings
          });
          // setRecentTasks(recent);
        } catch (userTasksError) {
          console.error('Error fetching user tasks:', userTasksError);
          const errorMsg = userTasksError.response?.data?.detail || 'Failed to fetch user tasks';
          setError(errorMsg);
          
          setStats({
            totalTasks: 0,
            pendingTasks: 0,
            completedTasks: 0,
            availableOfferings: offerings
          });
          // setRecentTasks([]);
        }
      } else {
        setError('User authentication required');
        setStats({
          totalTasks: 0,
          pendingTasks: 0,
          completedTasks: 0,
          availableOfferings: offerings
        });
        // setRecentTasks([]);
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
      // setRecentTasks([]);
    } finally {
      setLoading(false);
    }
  };

  const statCards = [
    {
      title: 'Total Tasks',
      value: stats.totalTasks,
      icon: <FileText size={24} />,
      bgColor: 'bg-gray-50',
      iconColor: 'text-black'
    },
    {
      title: 'Pending Tasks',
      value: stats.pendingTasks,
      icon: <Clock size={24} />,
      bgColor: 'bg-yellow-50',
      iconColor: 'text-yellow-600'
    },
    {
      title: 'Completed Tasks',
      value: stats.completedTasks,
      icon: <CheckCircle size={24} />,
      bgColor: 'bg-green-50',
      iconColor: 'text-green-600'
    },
    {
      title: 'Available Offerings',
      value: stats.availableOfferings,
      icon: <TrendingUp size={24} />,
      bgColor: 'bg-red-50',
      iconColor: 'text-red-600'
    }
  ];

  if (loading) {
    return <LoadingSpinner size="large" message="Loading dashboard..." />;
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-black mb-2">Welcome back, {user}</h2>
        <p className="text-gray-600">Here's an overview of your task processing activities</p>
      </div>

      {error && <ErrorMessage message={error} onClose={() => setError('')} />}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {statCards.map((stat, index) => (
          <motion.div
            key={stat.title}
            className={`${stat.bgColor} border-2 border-gray-200 p-6`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
          >
            <div className={`${stat.iconColor} mb-3`}>
              {stat.icon}
            </div>
            <div>
              <h3 className="text-3xl font-bold text-black">{stat.value}</h3>
              <p className="text-gray-600 text-sm mt-1">{stat.title}</p>
            </div>
          </motion.div>
        ))}
      </div>

      <div>
        <h3 className="text-xl font-bold text-black mb-4">Quick Actions</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <motion.div
            className="bg-white border-2 border-gray-200 p-6 cursor-pointer hover:border-red-600 transition-colors"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setActiveTab('submit')}
          >
            <h4 className="font-bold text-black mb-2 flex items-center">
              Submit New Task
              <ArrowRight className="ml-2 h-4 w-4 text-red-600" />
            </h4>
            <p className="text-gray-600 text-sm">Upload your data files and select processing tasks</p>
          </motion.div>
          
          <motion.div
            className="bg-white border-2 border-gray-200 p-6 cursor-pointer hover:border-red-600 transition-colors"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setActiveTab('status')}
          >
            <h4 className="font-bold text-black mb-2 flex items-center">
              View My Tasks
              <ArrowRight className="ml-2 h-4 w-4 text-red-600" />
            </h4>
            <p className="text-gray-600 text-sm">Track the status of your submitted tasks</p>
          </motion.div>
          
          <motion.div
            className="bg-white border-2 border-gray-200 p-6 cursor-pointer hover:border-red-600 transition-colors"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setActiveTab('offerings')}
          >
            <h4 className="font-bold text-black mb-2 flex items-center">
              Platform Offerings
              <ArrowRight className="ml-2 h-4 w-4 text-red-600" />
            </h4>
            <p className="text-gray-600 text-sm">Explore available data types and processing tasks</p>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default TicketDashboard;
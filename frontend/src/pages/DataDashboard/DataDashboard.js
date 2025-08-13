import React from 'react';
import { motion } from 'framer-motion';
import { BarChart3, TrendingUp, PieChart, Activity } from 'lucide-react';

const DataDashboard = () => {
  return (
    <div className="container section">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-4"
      >
        <h1 className="text-title">Data Dashboard</h1>
        <p className="text-body">Analytics and visualization for your processed data</p>
      </motion.div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="card">
          <div className="card-header">
            <BarChart3 size={24} />
            <h3 className="text-subheading">Task Analytics</h3>
          </div>
          <p className="text-body">Coming soon - Detailed analytics of your processed tasks</p>
        </div>

        <div className="card">
          <div className="card-header">
            <TrendingUp size={24} />
            <h3 className="text-subheading">Performance Metrics</h3>
          </div>
          <p className="text-body">Coming soon - Performance insights and trends</p>
        </div>

        <div className="card">
          <div className="card-header">
            <PieChart size={24} />
            <h3 className="text-subheading">Data Distribution</h3>
          </div>
          <p className="text-body">Coming soon - Visual breakdown of your data types</p>
        </div>

        <div className="card">
          <div className="card-header">
            <Activity size={24} />
            <h3 className="text-subheading">Real-time Monitoring</h3>
          </div>
          <p className="text-body">Coming soon - Live monitoring of processing activities</p>
        </div>
      </div>
    </div>
  );
};

export default DataDashboard;
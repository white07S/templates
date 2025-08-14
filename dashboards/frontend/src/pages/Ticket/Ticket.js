import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { LayoutDashboard, Plus, ListChecks, Briefcase } from 'lucide-react';
import TicketDashboard from './components/TicketDashboard';
import TaskSubmission from './components/TaskSubmission';
import TaskStatus from './components/TaskStatus';
import PlatformOfferings from './components/PlatformOfferings';

const Ticket = ({ user }) => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard size={18} /> },
    { id: 'submit', label: 'Submit New Task', icon: <Plus size={18} /> },
    { id: 'status', label: 'My Tasks', icon: <ListChecks size={18} /> },
    { id: 'offerings', label: 'Platform Offerings', icon: <Briefcase size={18} /> }
  ];

  const handleTaskSubmitted = () => {
    setRefreshTrigger(prev => prev + 1);
    setActiveTab('status');
  };

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-7xl mx-auto px-4 py-8">
        <motion.div 
          className="mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <h1 className="text-3xl font-bold text-black mb-2">Task Ticket Management</h1>
          <p className="text-gray-600">Submit and track your data processing tasks</p>
        </motion.div>

        <div className="flex space-x-1 border-b-2 border-gray-200 mb-6">
          {tabs.map((tab) => (
            <motion.button
              key={tab.id}
              className={`flex items-center px-4 py-3 font-medium transition-colors ${
                activeTab === tab.id 
                  ? 'text-red-600 border-b-2 border-red-600 -mb-0.5' 
                  : 'text-gray-600 hover:text-black'
              }`}
              onClick={() => setActiveTab(tab.id)}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <span className="mr-2">{tab.icon}</span>
              {tab.label}
            </motion.button>
          ))}
        </div>

        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="bg-white border-2 border-gray-200 p-6"
        >
          {activeTab === 'dashboard' && <TicketDashboard user={user} setActiveTab={setActiveTab} refreshTrigger={refreshTrigger} />}
          {activeTab === 'submit' && <TaskSubmission user={user} onTaskSubmitted={handleTaskSubmitted} />}
          {activeTab === 'status' && <TaskStatus user={user} refreshTrigger={refreshTrigger} />}
          {activeTab === 'offerings' && <PlatformOfferings />}
        </motion.div>
      </div>
    </div>
  );
};

export default Ticket;
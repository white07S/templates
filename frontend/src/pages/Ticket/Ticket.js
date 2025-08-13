import React, { useState } from 'react';
import { motion } from 'framer-motion';
import TicketDashboard from './components/TicketDashboard';
import TaskSubmission from './components/TaskSubmission';
import TaskStatus from './components/TaskStatus';
import PlatformOfferings from './components/PlatformOfferings';
import './Ticket.css';

const Ticket = ({ user }) => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const tabs = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'submit', label: 'Submit New Task' },
    { id: 'status', label: 'My Tasks' },
    { id: 'offerings', label: 'Platform Offerings' }
  ];

  const handleTaskSubmitted = () => {
    setRefreshTrigger(prev => prev + 1);
    setActiveTab('status');
  };

  return (
    <div className="ticket-page">
      <div className="container">
        <div className="ticket-header">
          <h1 className="text-title">Task Ticket Management</h1>
          <p className="text-body">Submit and track your data processing tasks</p>
        </div>

        <div className="ticket-tabs">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`tab-button ${activeTab === tab.id ? 'tab-active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <motion.div
          className="ticket-content"
          key={activeTab}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
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
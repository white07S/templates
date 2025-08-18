import React from 'react';
import { motion } from 'framer-motion';
import { MessageSquare, Bot, Users, Zap } from 'lucide-react';

const Chat = () => {
  return (
    <div className="container section">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-4"
      >
        <h1 className="text-title">AI Chat Interface</h1>
        <p className="text-body">Intelligent conversations about your data and analytics</p>
      </motion.div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="card">
          <div className="card-header">
            <Bot size={24} />
            <h3 className="text-subheading">AI Assistant</h3>
          </div>
          <p className="text-body">Coming soon - Smart AI assistant for data insights</p>
        </div>

        <div className="card">
          <div className="card-header">
            <MessageSquare size={24} />
            <h3 className="text-subheading">Query Interface</h3>
          </div>
          <p className="text-body">Coming soon - Natural language data queries</p>
        </div>

        <div className="card">
          <div className="card-header">
            <Users size={24} />
            <h3 className="text-subheading">Collaborative Chat</h3>
          </div>
          <p className="text-body">Coming soon - Team collaboration features</p>
        </div>

        <div className="card">
          <div className="card-header">
            <Zap size={24} />
            <h3 className="text-subheading">Quick Insights</h3>
          </div>
          <p className="text-body">Coming soon - Instant data insights and summaries</p>
        </div>
      </div>
    </div>
  );
};

export default Chat;
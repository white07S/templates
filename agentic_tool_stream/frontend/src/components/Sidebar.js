import React, { useState, useEffect } from 'react';
import { Search, MessageSquare, Plus } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const Sidebar = ({ sessions, currentSessionId, onSessionSelect, onNewSession }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredSessions, setFilteredSessions] = useState(sessions);

  useEffect(() => {
    if (searchQuery) {
      const filtered = sessions.filter(session => {
        const firstMessage = session.conversation?.[0]?.query || '';
        return firstMessage.toLowerCase().includes(searchQuery.toLowerCase());
      });
      setFilteredSessions(filtered);
    } else {
      setFilteredSessions(sessions);
    }
  }, [searchQuery, sessions]);

  const formatSessionName = (session) => {
    if (!session.conversation || session.conversation.length === 0) {
      return 'New Conversation';
    }
    const firstQuery = session.conversation[0].query;
    return firstQuery.length > 40 ? firstQuery.substring(0, 40) + '...' : firstQuery;
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    const now = new Date();
    const diffInHours = (now - date) / (1000 * 60 * 60);
    
    if (diffInHours < 24) {
      return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    } else if (diffInHours < 168) {
      return date.toLocaleDateString('en-US', { weekday: 'short' });
    } else {
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }
  };

  return (
    <div className="w-80 h-full bg-white border-r-2 border-black flex flex-col">
      <div className="p-4 border-b-2 border-black">
        <button
          onClick={onNewSession}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-red-600 text-white hover:bg-red-700 transition-colors"
        >
          <Plus size={20} />
          <span className="font-medium">New Chat</span>
        </button>
      </div>

      <div className="p-4 border-b-2 border-black">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500" size={18} />
          <input
            type="text"
            placeholder="Search conversations..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border-2 border-black focus:outline-none focus:border-red-600"
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        <AnimatePresence>
          {filteredSessions.map((session) => (
            <motion.div
              key={session.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
              onClick={() => onSessionSelect(session.id)}
              className={`p-4 border-b border-gray-200 cursor-pointer hover:bg-gray-50 transition-colors ${
                currentSessionId === session.id ? 'bg-red-50 border-l-4 border-l-red-600' : ''
              }`}
            >
              <div className="flex items-start gap-3">
                <MessageSquare size={18} className="text-gray-600 mt-1" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate">
                    {formatSessionName(session)}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    {formatTimestamp(session.conversation?.[0]?.timestamp)}
                  </p>
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {filteredSessions.length === 0 && (
          <div className="p-8 text-center text-gray-500">
            <p className="text-sm">No conversations found</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Sidebar;
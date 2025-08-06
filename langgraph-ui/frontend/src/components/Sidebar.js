import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Plus, 
  MessageCircle, 
  Calendar, 
  Clock, 
  Search, 
  X,
  Trash2,
  MoreVertical
} from 'lucide-react';
import { useChat } from '../contexts/ChatContext';

const Sidebar = ({ onClose }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const { 
    chatHistory, 
    currentChat, 
    createNewChat, 
    selectChat,
    loadChatHistory 
  } = useChat();

  const filteredChats = chatHistory.filter(chat => {
    if (!searchQuery.trim()) return true;
    
    const query = searchQuery.toLowerCase();
    return chat.messages?.some(msg => 
      msg.message?.toLowerCase().includes(query) || 
      msg.response?.toLowerCase().includes(query)
    );
  });

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 1) return 'Today';
    if (diffDays === 2) return 'Yesterday';
    if (diffDays <= 7) return `${diffDays} days ago`;
    return date.toLocaleDateString();
  };

  const getChatPreview = (chat) => {
    if (!chat.messages || chat.messages.length === 0) {
      return 'New conversation';
    }
    
    const firstMessage = chat.messages[0];
    const preview = firstMessage.message || firstMessage.response || '';
    return preview.length > 50 ? preview.substring(0, 50) + '...' : preview;
  };

  const handleNewChat = () => {
    createNewChat();
    if (onClose) onClose();
  };

  const handleSelectChat = (chat) => {
    selectChat(chat);
    if (onClose) onClose();
  };

  return (
    <div className="flex flex-col h-full bg-slate-800/95 backdrop-blur-sm border-r border-slate-700/50 w-full">
      {/* Header */}
      <div className="flex-shrink-0 p-4 border-b border-slate-700/50">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">Chat History</h2>
          <button
            onClick={onClose}
            className="md:hidden p-2 text-slate-400 hover:text-white rounded-lg hover:bg-slate-700/50 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* New Chat Button */}
        <motion.button
          onClick={handleNewChat}
          className="w-full flex items-center justify-center space-x-2 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-medium py-3 px-4 rounded-lg transition-all duration-200 mb-4"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <Plus className="w-5 h-5" />
          <span>New Chat</span>
        </motion.button>

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-4 h-4" />
          <input
            type="text"
            placeholder="Search conversations..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white placeholder-slate-400 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
          />
        </div>
      </div>

      {/* Chat List */}
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        <div className="p-2">
          {filteredChats.length === 0 ? (
            <div className="text-center py-8">
              {searchQuery ? (
                <div>
                  <Search className="w-12 h-12 text-slate-500 mx-auto mb-3" />
                  <p className="text-slate-400">No chats found</p>
                  <p className="text-slate-500 text-sm">Try a different search term</p>
                </div>
              ) : (
                <div>
                  <MessageCircle className="w-12 h-12 text-slate-500 mx-auto mb-3" />
                  <p className="text-slate-400">No conversations yet</p>
                  <p className="text-slate-500 text-sm">Start your first chat!</p>
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-1">
              {filteredChats.map((chat, index) => (
                <motion.div
                  key={chat.session_id || index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className={`group relative cursor-pointer rounded-lg p-3 transition-all duration-200 ${
                    currentChat?.sessionId === chat.session_id
                      ? 'bg-blue-500/20 border border-blue-500/30'
                      : 'hover:bg-slate-700/50 border border-transparent'
                  }`}
                  onClick={() => handleSelectChat(chat)}
                >
                  <div className="flex items-start space-x-3">
                    <div className="flex-shrink-0">
                      <div className="w-8 h-8 bg-gradient-to-br from-slate-600 to-slate-700 rounded-full flex items-center justify-center">
                        <MessageCircle className="w-4 h-4 text-slate-300" />
                      </div>
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-1">
                        <p className="text-sm font-medium text-white truncate">
                          {getChatPreview(chat)}
                        </p>
                        <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                          <button className="text-slate-400 hover:text-white p-1 rounded">
                            <MoreVertical className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-4 text-xs text-slate-400">
                        <div className="flex items-center space-x-1">
                          <Calendar className="w-3 h-3" />
                          <span>{formatDate(chat.updated_at)}</span>
                        </div>
                        {chat.messages && (
                          <div className="flex items-center space-x-1">
                            <MessageCircle className="w-3 h-3" />
                            <span>{chat.messages.length} msgs</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Active indicator */}
                  {currentChat?.sessionId === chat.session_id && (
                    <motion.div
                      layoutId="activeChat"
                      className="absolute left-0 top-1/2 transform -translate-y-1/2 w-1 h-8 bg-blue-500 rounded-r"
                    />
                  )}
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="flex-shrink-0 p-4 border-t border-slate-700/50">
        <div className="text-xs text-slate-500 text-center">
          <p>{filteredChats.length} conversation{filteredChats.length !== 1 ? 's' : ''}</p>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
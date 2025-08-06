import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, User, LogOut, Settings, MessageCircle } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { useChat } from '../contexts/ChatContext';

const Navbar = ({ onMenuClick }) => {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const { user, logout } = useAuth();
  const { createNewChat } = useChat();

  const handleLogout = () => {
    logout();
  };

  const getInitials = (name) => {
    return name
      .split(' ')
      .map(word => word[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  const getAvatarColor = (name) => {
    const colors = [
      'from-red-400 to-pink-500',
      'from-blue-400 to-indigo-500',
      'from-green-400 to-teal-500',
      'from-yellow-400 to-orange-500',
      'from-purple-400 to-pink-500',
      'from-indigo-400 to-purple-500'
    ];
    const index = name.length % colors.length;
    return colors[index];
  };

  return (
    <nav className="bg-slate-800/90 backdrop-blur-sm border-b border-slate-700/50 px-4 sm:px-6 lg:px-8 h-16">
      <div className="flex justify-between items-center h-full">
        {/* Left Side */}
        <div className="flex items-center">
          {/* Mobile menu button */}
          <motion.button
            onClick={onMenuClick}
            className="md:hidden p-2 rounded-lg text-slate-300 hover:text-white hover:bg-slate-700/50 transition-colors"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Menu className="w-6 h-6" />
          </motion.button>

          {/* Logo/Title */}
          <div className="flex items-center ml-2 md:ml-0">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center mr-3">
              <MessageCircle className="w-5 h-5 text-white" />
            </div>
            <div className="hidden sm:block">
              <h1 className="text-xl font-bold text-white">AI Assistant</h1>
              <p className="text-xs text-slate-400 -mt-1">Powered by LangChain</p>
            </div>
          </div>
        </div>

        {/* Right Side */}
        <div className="flex items-center space-x-4">
          {/* New Chat Button */}
          <motion.button
            onClick={createNewChat}
            className="hidden sm:flex items-center space-x-2 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-medium py-2 px-4 rounded-lg transition-all duration-200"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <MessageCircle className="w-4 h-4" />
            <span>New Chat</span>
          </motion.button>

          {/* User Profile Dropdown */}
          <div className="relative">
            <motion.button
              onClick={() => setDropdownOpen(!dropdownOpen)}
              className="flex items-center space-x-3 p-2 rounded-lg hover:bg-slate-700/50 transition-colors"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className={`w-8 h-8 bg-gradient-to-br ${getAvatarColor(user?.name)} rounded-full flex items-center justify-center`}>
                <span className="text-white text-sm font-medium">
                  {getInitials(user?.name || 'U')}
                </span>
              </div>
              <div className="hidden sm:block text-left">
                <p className="text-sm font-medium text-white">{user?.name}</p>
                <p className="text-xs text-slate-400">Online</p>
              </div>
            </motion.button>

            {/* Dropdown Menu */}
            <AnimatePresence>
              {dropdownOpen && (
                <>
                  <div
                    className="fixed inset-0 z-10"
                    onClick={() => setDropdownOpen(false)}
                  />
                  <motion.div
                    initial={{ opacity: 0, y: -10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -10, scale: 0.95 }}
                    transition={{ duration: 0.2 }}
                    className="absolute right-0 mt-2 w-56 bg-slate-800 rounded-lg shadow-xl border border-slate-700 py-1 z-20"
                  >
                    <div className="px-4 py-3 border-b border-slate-700">
                      <p className="text-sm font-medium text-white">{user?.name}</p>
                      <p className="text-xs text-slate-400">Signed in</p>
                    </div>
                    
                    <div className="py-1">
                      <button className="flex items-center w-full px-4 py-2 text-sm text-slate-300 hover:text-white hover:bg-slate-700/50 transition-colors">
                        <User className="w-4 h-4 mr-3" />
                        Profile
                      </button>
                      <button className="flex items-center w-full px-4 py-2 text-sm text-slate-300 hover:text-white hover:bg-slate-700/50 transition-colors">
                        <Settings className="w-4 h-4 mr-3" />
                        Settings
                      </button>
                    </div>
                    
                    <div className="border-t border-slate-700 py-1">
                      <button
                        onClick={handleLogout}
                        className="flex items-center w-full px-4 py-2 text-sm text-red-400 hover:text-red-300 hover:bg-slate-700/50 transition-colors"
                      >
                        <LogOut className="w-4 h-4 mr-3" />
                        Sign out
                      </button>
                    </div>
                  </motion.div>
                </>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
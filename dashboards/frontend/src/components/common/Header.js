import React from 'react';
import { Home } from 'lucide-react';
import { motion } from 'framer-motion';

const Header = () => {
  return (
    <motion.header 
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.3 }}
      className="bg-white border-b-2 border-red-600 shadow-md"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center cursor-pointer">
            <Home className="h-6 w-6 text-red-600 mr-3" />
            <h1 className="text-xl font-bold text-black">Task Ticket Platform</h1>
          </div>
        </div>
      </div>
    </motion.header>
  );
};

export default Header;
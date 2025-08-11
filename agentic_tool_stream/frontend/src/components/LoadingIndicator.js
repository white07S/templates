import React from 'react';
import { motion } from 'framer-motion';
import { Loader2 } from 'lucide-react';

const LoadingIndicator = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="flex items-center gap-3 px-6 py-4 bg-gray-50 border-t border-gray-200"
    >
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
      >
        <Loader2 size={20} className="text-red-600" />
      </motion.div>
      <span className="text-sm text-gray-600">
        Assistant is typing...
      </span>
    </motion.div>
  );
};

export default LoadingIndicator;
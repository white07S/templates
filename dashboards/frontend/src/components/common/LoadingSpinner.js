import React from 'react';
import { motion } from 'framer-motion';
import { Loader2 } from 'lucide-react';

const LoadingSpinner = ({ size = 'medium', message = '' }) => {
  const sizeClasses = {
    small: 'h-4 w-4',
    medium: 'h-8 w-8',
    large: 'h-12 w-12'
  };

  return (
    <motion.div 
      className="flex flex-col items-center justify-center p-8"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <Loader2 className={`${sizeClasses[size]} text-red-600 animate-spin`} />
      {message && <p className="mt-4 text-black text-center">{message}</p>}
    </motion.div>
  );
};

export default LoadingSpinner;
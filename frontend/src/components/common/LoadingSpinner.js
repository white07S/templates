import React from 'react';
import { motion } from 'framer-motion';
import './LoadingSpinner.css';

const LoadingSpinner = ({ size = 'medium', message = '' }) => {
  return (
    <motion.div 
      className="loading-spinner-container"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <div className={`loading-spinner loading-spinner-${size}`}>
        <div className="spinner-ring"></div>
      </div>
      {message && <p className="loading-message">{message}</p>}
    </motion.div>
  );
};

export default LoadingSpinner;
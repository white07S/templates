import React from 'react';
import { AlertCircle, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import './ErrorMessage.css';

const ErrorMessage = ({ 
  message, 
  onClose, 
  type = 'error',
  autoClose = false,
  autoCloseDelay = 5000 
}) => {
  React.useEffect(() => {
    if (autoClose && onClose) {
      const timer = setTimeout(onClose, autoCloseDelay);
      return () => clearTimeout(timer);
    }
  }, [autoClose, autoCloseDelay, onClose]);

  if (!message) return null;

  return (
    <AnimatePresence>
      <motion.div
        className={`error-message error-message-${type}`}
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.3 }}
      >
        <div className="error-message-content">
          <AlertCircle size={20} />
          <span className="error-message-text">{message}</span>
        </div>
        {onClose && (
          <button
            className="error-message-close"
            onClick={onClose}
            aria-label="Close"
          >
            <X size={18} />
          </button>
        )}
      </motion.div>
    </AnimatePresence>
  );
};

export default ErrorMessage;
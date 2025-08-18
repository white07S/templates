import React from 'react';
import { AlertCircle, X, CheckCircle, Info } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

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

  const typeStyles = {
    error: 'bg-red-50 border-red-600 text-red-800',
    success: 'bg-green-50 border-green-600 text-green-800',
    warning: 'bg-yellow-50 border-yellow-600 text-yellow-800',
    info: 'bg-blue-50 border-blue-600 text-blue-800'
  };

  const icons = {
    error: <AlertCircle size={20} />,
    success: <CheckCircle size={20} />,
    warning: <AlertCircle size={20} />,
    info: <Info size={20} />
  };

  return (
    <AnimatePresence>
      <motion.div
        className={`flex items-center justify-between p-4 border-l-4 ${typeStyles[type]} mb-4`}
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: -20 }}
        transition={{ duration: 0.3 }}
      >
        <div className="flex items-center">
          <span className="mr-3">{icons[type]}</span>
          <span className="font-medium">{message}</span>
        </div>
        {onClose && (
          <button
            className="ml-4 p-1 hover:bg-white hover:bg-opacity-50 transition-colors"
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
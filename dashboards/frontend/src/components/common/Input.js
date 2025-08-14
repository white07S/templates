import React from 'react';
import { motion } from 'framer-motion';

const Input = ({
  label,
  error,
  helperText,
  required = false,
  className = '',
  id,
  ...props
}) => {
  const inputId = id || `input-${Math.random().toString(36).substr(2, 9)}`;
  
  return (
    <motion.div 
      className={`mb-4 ${className}`}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {label && (
        <label htmlFor={inputId} className="block text-sm font-medium text-black mb-1">
          {label}
          {required && <span className="text-red-600 ml-1">*</span>}
        </label>
      )}
      <input
        id={inputId}
        className={`w-full px-3 py-2 border-2 ${error ? 'border-red-600' : 'border-gray-300'} bg-white text-black focus:outline-none focus:border-red-600 transition-colors`}
        {...props}
      />
      {(error || helperText) && (
        <span className={`text-sm mt-1 block ${error ? 'text-red-600' : 'text-gray-600'}`}>
          {error || helperText}
        </span>
      )}
    </motion.div>
  );
};

export default Input;
import React from 'react';
import { motion } from 'framer-motion';
import './Button.css';

const Button = ({ 
  children, 
  onClick, 
  type = 'button', 
  variant = 'primary', 
  size = 'medium',
  disabled = false,
  loading = false,
  className = '',
  icon = null,
  ...props 
}) => {
  const buttonClass = `
    btn 
    btn-${variant} 
    btn-${size}
    ${disabled || loading ? 'btn-disabled' : ''}
    ${className}
  `.trim();

  return (
    <motion.button
      type={type}
      className={buttonClass}
      onClick={onClick}
      disabled={disabled || loading}
      whileHover={!disabled && !loading ? { scale: 1.02 } : {}}
      whileTap={!disabled && !loading ? { scale: 0.98 } : {}}
      transition={{ duration: 0.2 }}
      {...props}
    >
      {loading ? (
        <span className="btn-loading">
          <span className="spinner"></span>
          <span>Loading...</span>
        </span>
      ) : (
        <>
          {icon && <span className="btn-icon">{icon}</span>}
          {children}
        </>
      )}
    </motion.button>
  );
};

export default Button;
import React from 'react';
import './Input.css';

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
    <div className={`input-wrapper ${className}`}>
      {label && (
        <label htmlFor={inputId} className="input-label">
          {label}
          {required && <span className="input-required">*</span>}
        </label>
      )}
      <input
        id={inputId}
        className={`input ${error ? 'input-error' : ''}`}
        {...props}
      />
      {(error || helperText) && (
        <span className={`input-helper ${error ? 'input-helper-error' : ''}`}>
          {error || helperText}
        </span>
      )}
    </div>
  );
};

export default Input;
import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Loader2, CheckCircle, AlertCircle, Terminal } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export const Tool = ({ children, defaultOpen = false }) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="mb-2 border border-gray-300 rounded-sm overflow-hidden">
      {React.Children.map(children, child => {
        if (child?.type === ToolHeader) {
          return React.cloneElement(child, { 
            isOpen, 
            onToggle: () => setIsOpen(!isOpen) 
          });
        }
        if (child?.type === ToolContent) {
          return (
            <AnimatePresence>
              {isOpen && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  {child}
                </motion.div>
              )}
            </AnimatePresence>
          );
        }
        return child;
      })}
    </div>
  );
};

export const ToolHeader = ({ type, state, isOpen, onToggle }) => {
  const getStateIcon = () => {
    switch (state) {
      case 'input-streaming':
      case 'running':
        return <Loader2 className="w-4 h-4 animate-spin text-blue-600" />;
      case 'output-available':
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'output-error':
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-600" />;
      default:
        return <Terminal className="w-4 h-4 text-gray-600" />;
    }
  };

  const getStateText = () => {
    switch (state) {
      case 'input-streaming':
        return 'Processing input...';
      case 'running':
        return 'Running...';
      case 'output-available':
      case 'completed':
        return 'Completed';
      case 'output-error':
      case 'error':
        return 'Error';
      default:
        return 'Ready';
    }
  };

  return (
    <div 
      className="flex items-center justify-between p-3 bg-gray-50 hover:bg-gray-100 cursor-pointer transition-colors"
      onClick={onToggle}
    >
      <div className="flex items-center gap-2">
        {isOpen ? (
          <ChevronDown className="w-4 h-4 text-gray-600" />
        ) : (
          <ChevronRight className="w-4 h-4 text-gray-600" />
        )}
        {getStateIcon()}
        <span className="font-mono text-sm font-semibold text-gray-900">{type}</span>
      </div>
      <span className="text-xs text-gray-600">{getStateText()}</span>
    </div>
  );
};

export const ToolContent = ({ children }) => {
  return (
    <div className="border-t border-gray-200">
      {children}
    </div>
  );
};

export const ToolInput = ({ input }) => {
  const formatInput = () => {
    if (typeof input === 'object') {
      return JSON.stringify(input, null, 2);
    }
    return input;
  };

  return (
    <div className="p-3 bg-gray-50">
      <div className="text-xs font-semibold text-gray-600 mb-2">Input:</div>
      <pre className="text-xs bg-white border border-gray-200 p-2 rounded overflow-x-auto">
        <code>{formatInput()}</code>
      </pre>
    </div>
  );
};

export const ToolOutput = ({ output, error }) => {
  const formatOutput = () => {
    if (typeof output === 'object') {
      return JSON.stringify(output, null, 2);
    }
    return output;
  };

  return (
    <div className="p-3">
      <div className="text-xs font-semibold text-gray-600 mb-2">
        {error ? 'Error:' : 'Output:'}
      </div>
      <pre className={`text-xs p-2 rounded overflow-x-auto ${
        error ? 'bg-red-50 border border-red-200 text-red-800' : 'bg-white border border-gray-200'
      }`}>
        <code>{error || formatOutput()}</code>
      </pre>
    </div>
  );
};

export default Tool;
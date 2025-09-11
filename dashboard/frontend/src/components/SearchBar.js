import React, { useState, useCallback } from 'react';
import { Search, X } from 'lucide-react';
import { useDashboard } from '../contexts/DashboardContext';
import { motion } from 'framer-motion';

const SearchBar = () => {
  const { state, actions } = useDashboard();
  const [localQuery, setLocalQuery] = useState(state.searchQuery);

  // Debounced search function
  const debouncedSearch = useCallback(
    debounce((query) => {
      actions.setSearchQuery(query);
    }, 300),
    [actions]
  );

  const handleInputChange = (e) => {
    const value = e.target.value;
    setLocalQuery(value);
    debouncedSearch(value);
  };

  const clearSearch = () => {
    setLocalQuery('');
    actions.setSearchQuery('');
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    actions.setSearchQuery(localQuery);
  };

  return (
    <motion.form 
      onSubmit={handleSubmit}
      className="relative w-full"
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
    >
      <div className="relative">
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <Search className="h-5 w-5 text-gray-400" />
        </div>
        
        <input
          type="text"
          value={localQuery}
          onChange={handleInputChange}
          placeholder="Search descriptions, AI taxonomy, or ERMS taxonomy..."
          className="
            w-full pl-10 pr-12 py-3 border border-gray-300  
            focus:ring-2 focus:ring-primary-500 focus:border-transparent
            text-gray-900 placeholder-gray-500 bg-white
            transition-all duration-200
          "
        />
        
        {localQuery && (
          <button
            type="button"
            onClick={clearSearch}
            className="
              absolute inset-y-0 right-0 pr-3 flex items-center
              text-gray-400 hover:text-gray-600 transition-colors
            "
          >
            <X className="h-5 w-5" />
          </button>
        )}
      </div>
      
      {state.loading && (
        <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
          <div className="animate-spin -full h-5 w-5 border-2 border-primary-500 border-t-transparent"></div>
        </div>
      )}
      
      {localQuery && (
        <motion.div 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="absolute top-full mt-2 left-0 right-0 bg-white border border-gray-200  shadow-lg p-3 z-10"
        >
          <div className="text-sm text-gray-600">
            Searching in: <span className="font-medium">{state.selectedDataset.replace('_', ' ')}</span>
          </div>
          {state.searchQuery !== localQuery && (
            <div className="text-xs text-gray-500 mt-1">
              Press Enter to search immediately
            </div>
          )}
        </motion.div>
      )}
    </motion.form>
  );
};

// Debounce utility function
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

export default SearchBar;
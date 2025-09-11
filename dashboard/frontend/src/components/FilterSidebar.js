import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Filter, ChevronDown, ChevronRight, X } from 'lucide-react';
import { useDashboard } from '../contexts/DashboardContext';

const FilterSection = ({ title, items, selectedValue, onSelect, placeholder }) => {
  const [isOpen, setIsOpen] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');

  const filteredItems = items.filter(item => 
    item.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="border-b border-gray-200 pb-4 mb-4">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between w-full text-left font-medium text-gray-900 hover:text-primary-600 transition-colors"
      >
        <span>{title}</span>
        {isOpen ? (
          <ChevronDown className="h-4 w-4" />
        ) : (
          <ChevronRight className="h-4 w-4" />
        )}
      </button>
      
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="mt-3 space-y-2"
          >
            {/* Search input for filtering */}
            {items.length > 5 && (
              <input
                type="text"
                placeholder={`Search ${title.toLowerCase()}...`}
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-gray-300  focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            )}

            {/* Clear selection option */}
            {selectedValue && (
              <button
                onClick={() => onSelect('')}
                className="flex items-center w-full text-left text-sm text-gray-600 hover:text-primary-600 hover:bg-gray-50 px-2 py-1  transition-colors"
              >
                <X className="h-3 w-3 mr-2" />
                Clear {title}
              </button>
            )}

            {/* Filter options */}
            <div className="max-h-48 overflow-y-auto">
              {filteredItems.length > 0 ? (
                filteredItems.map((item, index) => (
                  <button
                    key={`${item}-${index}`}
                    onClick={() => onSelect(item)}
                    className={`
                      block w-full text-left text-sm px-2 py-1  transition-colors
                      ${selectedValue === item
                        ? 'bg-primary-100 text-primary-800 font-medium'
                        : 'text-gray-700 hover:bg-gray-50 hover:text-primary-600'
                      }
                    `}
                  >
                    {item}
                  </button>
                ))
              ) : (
                <div className="text-sm text-gray-500 px-2 py-1">
                  No items found
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const FilterSidebar = () => {
  const { state, actions } = useDashboard();

  const hasActiveFilters = state.selectedAiTaxonomy || state.selectedErmsTaxonomy;

  const clearAllFilters = () => {
    actions.setAiTaxonomy('');
    actions.setErmsTaxonomy('');
  };

  return (
    <motion.div 
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className="w-80 bg-white border-r border-gray-200 h-screen overflow-y-auto"
    >
      <div className="p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center">
            <Filter className="h-5 w-5 text-primary-600 mr-2" />
            <h2 className="text-lg font-semibold text-gray-900">Filters</h2>
          </div>
          {hasActiveFilters && (
            <button
              onClick={clearAllFilters}
              className="text-sm text-primary-600 hover:text-primary-700 font-medium transition-colors"
            >
              Clear All
            </button>
          )}
        </div>

        {/* Active Dataset Display */}
        <div className="mb-6 p-4 bg-gray-50 ">
          <div className="text-sm font-medium text-gray-700 mb-1">Active Dataset</div>
          <div className="text-lg font-semibold text-gray-900 capitalize">
            {state.selectedDataset.replace('_', ' ')}
          </div>
          <div className="text-sm text-gray-500 mt-1">
            {state.totalRecords} total records
          </div>
        </div>

        {/* Filter Sections */}
        {state.taxonomies.ai_taxonomies && state.taxonomies.ai_taxonomies.length > 0 && (
          <FilterSection
            title="AI Taxonomy"
            items={state.taxonomies.ai_taxonomies}
            selectedValue={state.selectedAiTaxonomy}
            onSelect={actions.setAiTaxonomy}
            placeholder="Select AI taxonomy..."
          />
        )}

        {state.taxonomies.erms_taxonomies && state.taxonomies.erms_taxonomies.length > 0 && (
          <FilterSection
            title="ERMS Taxonomy"
            items={state.taxonomies.erms_taxonomies}
            selectedValue={state.selectedErmsTaxonomy}
            onSelect={actions.setErmsTaxonomy}
            placeholder="Select ERMS taxonomy..."
          />
        )}

        {/* Filter Summary */}
        {hasActiveFilters && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-6 p-4 bg-primary-50 border border-primary-200 "
          >
            <div className="text-sm font-medium text-primary-800 mb-2">Active Filters</div>
            <div className="space-y-1">
              {state.selectedAiTaxonomy && (
                <div className="flex items-center justify-between">
                  <span className="text-xs text-primary-700">AI: {state.selectedAiTaxonomy}</span>
                  <button
                    onClick={() => actions.setAiTaxonomy('')}
                    className="text-primary-600 hover:text-primary-800"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </div>
              )}
              {state.selectedErmsTaxonomy && (
                <div className="flex items-center justify-between">
                  <span className="text-xs text-primary-700">ERMS: {state.selectedErmsTaxonomy}</span>
                  <button
                    onClick={() => actions.setErmsTaxonomy('')}
                    className="text-primary-600 hover:text-primary-800"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </div>
              )}
            </div>
          </motion.div>
        )}

        {/* Loading State */}
        {state.loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-6 flex items-center justify-center py-4"
          >
            <div className="animate-spin  h-6 w-6 border-2 border-primary-500 border-t-transparent"></div>
            <span className="ml-2 text-sm text-gray-600">Loading...</span>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
};

export default FilterSidebar;
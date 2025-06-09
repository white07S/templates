import React, { useState, useEffect } from 'react';

const FiltersPanel = ({ filters, onFilterChange, onClearFilters }) => {
  const [filterOptions, setFilterOptions] = useState({});
  const [loading, setLoading] = useState(true);
  const [expandedSections, setExpandedSections] = useState({
    amount: true,
    date: true,
    business: true,
    geography: true,
    risk: true
  });

  useEffect(() => {
    fetchFilterOptions();
  }, []);

  const fetchFilterOptions = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/filter-options');
      if (!response.ok) {
        throw new Error('Failed to fetch filter options');
      }
      const data = await response.json();
      setFilterOptions(data);
    } catch (error) {
      console.error('Error fetching filter options:', error);
    } finally {
      setLoading(false);
    }
  };

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const handleMultiSelectChange = (field, value) => {
    const currentValues = filters[field] || [];
    const newValues = currentValues.includes(value)
      ? currentValues.filter(v => v !== value)
      : [...currentValues, value];
    onFilterChange({ [field]: newValues });
  };

  const getActiveFiltersCount = () => {
    let count = 0;
    if (filters.search) count++;
    if (filters.loss_amount_min !== null || filters.loss_amount_max !== null) count++;
    if (filters.date_from || filters.date_to) count++;
    ['business_lines', 'regions', 'countries', 'risk_categories', 'nfr_taxonomies', 'ubs_divisions'].forEach(field => {
      if (filters[field] && filters[field].length > 0) count++;
    });
    return count;
  };

  const FilterSection = ({ title, section, children }) => (
    <div className="border-b border-gray-200 pb-4 mb-4">
      <button
        onClick={() => toggleSection(section)}
        className="flex items-center justify-between w-full text-left"
      >
        <h3 className="text-sm font-medium text-gray-900">{title}</h3>
        <svg
          className={`w-4 h-4 transform transition-transform ${
            expandedSections[section] ? 'rotate-180' : ''
          }`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {expandedSections[section] && (
        <div className="mt-3 space-y-3">
          {children}
        </div>
      )}
    </div>
  );

  const MultiSelect = ({ options, selectedValues, onChange, placeholder }) => (
    <div className="space-y-2">
      <div className="max-h-40 overflow-y-auto border border-gray-300 rounded-md bg-white">
        {options.map((option) => (
          <label
            key={option}
            className="flex items-center px-3 py-2 hover:bg-gray-50 cursor-pointer"
          >
            <input
              type="checkbox"
              checked={selectedValues.includes(option)}
              onChange={() => onChange(option)}
              className="mr-2 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="text-sm text-gray-700 truncate">{option}</span>
          </label>
        ))}
      </div>
      {selectedValues.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selectedValues.map((value) => (
            <span
              key={value}
              className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
            >
              {value}
              <button
                onClick={() => onChange(value)}
                className="ml-1 text-blue-600 hover:text-blue-800"
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  );

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-300 rounded w-24 mb-4"></div>
          <div className="space-y-3">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-10 bg-gray-300 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow">
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-medium text-gray-900">Filters</h2>
          {getActiveFiltersCount() > 0 && (
            <button
              onClick={onClearFilters}
              className="text-sm text-blue-600 hover:text-blue-800 font-medium"
            >
              Clear All ({getActiveFiltersCount()})
            </button>
          )}
        </div>

        <div className="space-y-6">
          {/* Loss Amount Range */}
          <FilterSection title="Loss Amount ($M)" section="amount">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  Min Amount
                </label>
                <input
                  type="number"
                  value={filters.loss_amount_min || ''}
                  onChange={(e) => onFilterChange({ 
                    loss_amount_min: e.target.value ? parseFloat(e.target.value) : null 
                  })}
                  placeholder="0"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  Max Amount
                </label>
                <input
                  type="number"
                  value={filters.loss_amount_max || ''}
                  onChange={(e) => onFilterChange({ 
                    loss_amount_max: e.target.value ? parseFloat(e.target.value) : null 
                  })}
                  placeholder="100"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
            </div>
            {filterOptions.loss_amount_range && (
              <div className="text-xs text-gray-500">
                Range: ${filterOptions.loss_amount_range.min}M - ${filterOptions.loss_amount_range.max}M
              </div>
            )}
          </FilterSection>

          {/* Date Range */}
          <FilterSection title="Date Range" section="date">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  From
                </label>
                <input
                  type="date"
                  value={filters.date_from || ''}
                  onChange={(e) => onFilterChange({ date_from: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  To
                </label>
                <input
                  type="date"
                  value={filters.date_to || ''}
                  onChange={(e) => onFilterChange({ date_to: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
            </div>
          </FilterSection>

          {/* Business Lines */}
          <FilterSection title="Business Lines" section="business">
            <MultiSelect
              options={filterOptions.business_lines || []}
              selectedValues={filters.business_lines || []}
              onChange={(value) => handleMultiSelectChange('business_lines', value)}
              placeholder="Select business lines"
            />
          </FilterSection>

          {/* Geography */}
          <FilterSection title="Geography" section="geography">
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-2">
                  Regions
                </label>
                <MultiSelect
                  options={filterOptions.regions || []}
                  selectedValues={filters.regions || []}
                  onChange={(value) => handleMultiSelectChange('regions', value)}
                  placeholder="Select regions"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-2">
                  Countries
                </label>
                <MultiSelect
                  options={filterOptions.countries || []}
                  selectedValues={filters.countries || []}
                  onChange={(value) => handleMultiSelectChange('countries', value)}
                  placeholder="Select countries"
                />
              </div>
            </div>
          </FilterSection>

          {/* Risk Categories */}
          <FilterSection title="Risk & Taxonomy" section="risk">
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-2">
                  Risk Categories
                </label>
                <MultiSelect
                  options={filterOptions.risk_categories || []}
                  selectedValues={filters.risk_categories || []}
                  onChange={(value) => handleMultiSelectChange('risk_categories', value)}
                  placeholder="Select risk categories"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-2">
                  NFR Taxonomies
                </label>
                <MultiSelect
                  options={filterOptions.nfr_taxonomies || []}
                  selectedValues={filters.nfr_taxonomies || []}
                  onChange={(value) => handleMultiSelectChange('nfr_taxonomies', value)}
                  placeholder="Select taxonomies"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-2">
                  UBS Divisions
                </label>
                <MultiSelect
                  options={filterOptions.ubs_divisions || []}
                  selectedValues={filters.ubs_divisions || []}
                  onChange={(value) => handleMultiSelectChange('ubs_divisions', value)}
                  placeholder="Select divisions"
                />
              </div>
            </div>
          </FilterSection>
        </div>
      </div>
    </div>
  );
};

export default FiltersPanel;
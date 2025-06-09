import React, { useState, useRef, useEffect } from 'react';

const SearchBar = ({ value, onChange, placeholder = "Search across all fields..." }) => {
  const [isFocused, setIsFocused] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const inputRef = useRef(null);
  const debounceRef = useRef(null);

  // Predefined search suggestions based on common terms
  const commonSearchTerms = [
    'fraud', 'cybersecurity', 'data privacy', 'process failures',
    'external fraud', 'internal fraud', 'execution errors',
    'corporate finance', 'retail banking', 'wealth management',
    'germany', 'usa', 'uk', 'singapore',
    'severe impact', 'moderate impact', 'minimal impact'
  ];

  useEffect(() => {
    // Clear debounce on unmount
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, []);

  const handleInputChange = (e) => {
    const newValue = e.target.value;
    onChange(newValue);

    // Debounce suggestions
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    debounceRef.current = setTimeout(() => {
      if (newValue.length > 0) {
        const filtered = commonSearchTerms.filter(term =>
          term.toLowerCase().includes(newValue.toLowerCase())
        ).slice(0, 5);
        setSuggestions(filtered);
        setShowSuggestions(filtered.length > 0);
      } else {
        setSuggestions([]);
        setShowSuggestions(false);
      }
    }, 200);
  };

  const handleSuggestionClick = (suggestion) => {
    onChange(suggestion);
    setShowSuggestions(false);
    inputRef.current.focus();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Escape') {
      setShowSuggestions(false);
      inputRef.current.blur();
    } else if (e.key === 'Enter') {
      setShowSuggestions(false);
    }
  };

  const handleFocus = () => {
    setIsFocused(true);
    if (value.length > 0 && suggestions.length > 0) {
      setShowSuggestions(true);
    }
  };

  const handleBlur = () => {
    setIsFocused(false);
    // Delay hiding suggestions to allow clicking
    setTimeout(() => {
      setShowSuggestions(false);
    }, 200);
  };

  const clearSearch = () => {
    onChange('');
    setShowSuggestions(false);
    inputRef.current.focus();
  };

  return (
    <div className="relative">
      <div className={`relative transition-all duration-200 ${
        isFocused ? 'transform scale-105' : ''
      }`}>
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <svg
            className={`h-5 w-5 transition-colors duration-200 ${
              isFocused ? 'text-blue-500' : 'text-gray-400'
            }`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
        </div>
        
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={handleInputChange}
          onFocus={handleFocus}
          onBlur={handleBlur}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className={`block w-full pl-10 pr-10 py-2 border rounded-lg text-sm transition-all duration-200 focus:outline-none ${
            isFocused
              ? 'border-blue-500 ring-2 ring-blue-200 bg-white shadow-lg'
              : 'border-gray-300 bg-gray-50 hover:bg-white hover:border-gray-400'
          }`}
          style={{ minWidth: '300px' }}
        />
        
        {value && (
          <button
            onClick={clearSearch}
            className="absolute inset-y-0 right-0 pr-3 flex items-center"
          >
            <svg
              className="h-4 w-4 text-gray-400 hover:text-gray-600 transition-colors duration-200"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        )}
      </div>

      {/* Search Suggestions */}
      {showSuggestions && suggestions.length > 0 && (
        <div className="absolute z-50 top-full left-0 right-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto">
          <div className="py-1">
            {suggestions.map((suggestion, index) => (
              <button
                key={index}
                onClick={() => handleSuggestionClick(suggestion)}
                className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-blue-50 hover:text-blue-700 transition-colors duration-150 flex items-center"
              >
                <svg className="w-4 h-4 mr-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <span>{suggestion}</span>
              </button>
            ))}
          </div>
          
          {value && !suggestions.some(s => s.toLowerCase() === value.toLowerCase()) && (
            <div className="border-t border-gray-100 px-4 py-2 bg-gray-50">
              <div className="flex items-center text-xs text-gray-500">
                <svg className="w-3 h-3 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Press Enter to search for "{value}"
              </div>
            </div>
          )}
        </div>
      )}

      {/* Search Tips (shown when focused and no value) */}
      {isFocused && !value && (
        <div className="absolute z-50 top-full left-0 right-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg">
          <div className="p-4">
            <h4 className="text-sm font-medium text-gray-900 mb-2">Search Tips</h4>
            <ul className="text-xs text-gray-600 space-y-1">
              <li>• Search across descriptions, summaries, and company names</li>
              <li>• Use keywords like "fraud", "cybersecurity", or company names</li>
              <li>• Results update automatically as you type</li>
            </ul>
            
            <div className="mt-3 pt-3 border-t border-gray-100">
              <h5 className="text-xs font-medium text-gray-700 mb-2">Quick Searches</h5>
              <div className="flex flex-wrap gap-1">
                {commonSearchTerms.slice(0, 4).map((term) => (
                  <button
                    key={term}
                    onClick={() => handleSuggestionClick(term)}
                    className="px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded hover:bg-blue-100 hover:text-blue-700 transition-colors duration-150"
                  >
                    {term}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SearchBar;
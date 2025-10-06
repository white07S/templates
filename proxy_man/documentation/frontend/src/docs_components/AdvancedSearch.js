import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Search, X, FileText, Hash, Code, ArrowRight, Command } from 'lucide-react';
import { searchAPI } from '../components/api';

const AdvancedSearch = ({ isOpen, onClose, onNavigate }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [results, setResults] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [isSearching, setIsSearching] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const searchInputRef = useRef(null);
  const searchTimeoutRef = useRef(null);

  // Categories for filtering
  const categories = [
    { id: 'all', label: 'All', icon: null },
    { id: 'heading', label: 'Headings', icon: Hash },
    { id: 'content', label: 'Content', icon: FileText },
    { id: 'code', label: 'Code', icon: Code },
  ];

  // Focus search input when modal opens
  useEffect(() => {
    if (isOpen && searchInputRef.current) {
      searchInputRef.current.focus();
      setSearchTerm('');
      setResults([]);
      setSelectedIndex(0);
    }
  }, [isOpen]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (!isOpen) return;

      if (e.key === 'Escape') {
        onClose();
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex(prev => Math.min(prev + 1, results.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex(prev => Math.max(prev - 1, 0));
      } else if (e.key === 'Enter' && results[selectedIndex]) {
        e.preventDefault();
        handleResultClick(results[selectedIndex]);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, results, selectedIndex, onClose]);

  // Debounced search
  const performSearch = useCallback(async (query, category) => {
    if (!query.trim()) {
      setResults([]);
      return;
    }

    setIsSearching(true);
    try {
      const data = await searchAPI.search(query, category);
      setResults(data);
      setSelectedIndex(0);
    } catch (error) {
      console.error('Search failed:', error);
      setResults([]);
    } finally {
      setIsSearching(false);
    }
  }, []);

  // Handle search input change with debouncing
  const handleSearchChange = (value) => {
    setSearchTerm(value);

    // Clear existing timeout
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }

    // Set new timeout for debounced search
    searchTimeoutRef.current = setTimeout(() => {
      performSearch(value, selectedCategory);
    }, 300);
  };

  // Handle category change
  const handleCategoryChange = (category) => {
    setSelectedCategory(category);
    if (searchTerm) {
      performSearch(searchTerm, category);
    }
  };

  // Handle result click - navigate to the page and section
  const handleResultClick = (result) => {
    // Extract base page ID and anchor
    const [baseId] = result.id.split('#');
    const pageData = {
      id: baseId,
      title: result.title,
      path: result.path,
      anchor: result.anchor
    };

    onNavigate(pageData);
    onClose();
  };

  // Highlight search term in text
  const highlightText = (text, term) => {
    if (!term) return text;

    const parts = text.split(new RegExp(`(${term})`, 'gi'));
    return parts.map((part, index) =>
      part.toLowerCase() === term.toLowerCase() ? (
        <mark key={index} className="bg-yellow-200 text-black px-0.5">
          {part}
        </mark>
      ) : (
        part
      )
    );
  };

  // Get icon for result category
  const getCategoryIcon = (category) => {
    switch (category) {
      case 'heading':
        return <Hash className="w-4 h-4 text-gray-500" />;
      case 'code':
        return <Code className="w-4 h-4 text-gray-500" />;
      default:
        return <FileText className="w-4 h-4 text-gray-500" />;
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black bg-opacity-50 transition-opacity"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="fixed inset-x-0 top-16 mx-auto max-w-4xl">
        <div className="bg-white border-2 border-gray-300 shadow-2xl overflow-hidden mx-4">
          {/* Search Header */}
          <div className="border-b-2 border-gray-200 bg-gray-50">
            <div className="flex items-center px-6 py-4">
              <Search className="w-6 h-6 text-gray-500 mr-4" />
              <input
                ref={searchInputRef}
                type="text"
                value={searchTerm}
                onChange={(e) => handleSearchChange(e.target.value)}
                placeholder="Search documentation..."
                className="flex-1 outline-none text-xl bg-transparent font-medium"
              />
              <button
                onClick={onClose}
                className="p-2 hover:bg-gray-200 transition-colors rounded"
              >
                <X className="w-6 h-6 text-gray-600" />
              </button>
            </div>

            {/* Category Filters */}
            <div className="flex items-center px-6 py-3 border-t border-gray-100 bg-white">
              {categories.map((cat) => (
                <button
                  key={cat.id}
                  onClick={() => handleCategoryChange(cat.id)}
                  className={`
                    flex items-center px-4 py-2 mr-3 text-base font-medium
                    transition-colors border-2
                    ${selectedCategory === cat.id
                      ? 'bg-gray-50 text-gray-700 border-gray-400'
                      : 'bg-white text-gray-600 border-gray-200 hover:bg-gray-50'
                    }
                  `}
                >
                  {cat.icon && <cat.icon className="w-4 h-4 mr-2" />}
                  {cat.label}
                </button>
              ))}
            </div>
          </div>

          {/* Search Results */}
          <div className="max-h-[500px] overflow-y-auto">
            {isSearching ? (
              <div className="p-12 text-center text-gray-500">
                <div className="text-lg">Searching...</div>
              </div>
            ) : results.length === 0 && searchTerm ? (
              <div className="p-12 text-center text-gray-500">
                <div className="text-lg">No results found for "{searchTerm}"</div>
              </div>
            ) : results.length === 0 ? (
              <div className="p-12 text-center text-gray-400">
                <div className="text-lg mb-3">Start typing to search</div>
                <div className="text-sm">
                  <kbd className="px-2 py-1 bg-gray-100 border border-gray-200">↑</kbd>
                  <kbd className="px-2 py-1 bg-gray-100 border border-gray-200 mx-1">↓</kbd>
                  to navigate,
                  <kbd className="px-2 py-1 bg-gray-100 border border-gray-200 ml-1">Enter</kbd>
                  to select,
                  <kbd className="px-2 py-1 bg-gray-100 border border-gray-200 ml-1">Esc</kbd>
                  to close
                </div>
              </div>
            ) : (
              <div>
                {results.map((result, index) => (
                  <button
                    key={result.id}
                    onClick={() => handleResultClick(result)}
                    onMouseEnter={() => setSelectedIndex(index)}
                    className={`
                      w-full px-6 py-4 text-left border-b border-gray-100
                      transition-colors group
                      ${index === selectedIndex ? 'bg-gray-50' : 'hover:bg-gray-50'}
                    `}
                  >
                    <div className="flex items-start">
                      <div className="mr-3 mt-1">
                        {getCategoryIcon(result.category)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center mb-2">
                          <span className="text-base font-semibold text-gray-900">
                            {result.title}
                          </span>
                          {result.heading && (
                            <>
                              <ArrowRight className="w-4 h-4 mx-2 text-gray-400" />
                              <span className="text-base text-gray-700">
                                {result.heading}
                              </span>
                            </>
                          )}
                        </div>
                        <div className="text-base text-gray-600 line-clamp-2">
                          {highlightText(result.snippet, searchTerm)}
                        </div>
                        <div className="flex items-center mt-2 text-sm text-gray-500">
                          <span className="px-1.5 py-0.5 bg-gray-100 mr-2">
                            {result.category}
                          </span>
                          <span>{result.path}</span>
                        </div>
                      </div>
                      {index === selectedIndex && (
                        <div className="ml-2 opacity-0 group-hover:opacity-100 transition-opacity">
                          <ArrowRight className="w-4 h-4 text-gray-400" />
                        </div>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="px-4 py-2 border-t border-gray-200 bg-gray-50">
            <div className="flex items-center justify-between text-xs text-gray-500">
              <div className="flex items-center">
                <Command className="w-3 h-3 mr-1" />
                <span>K to open search</span>
              </div>
              <div>
                {results.length > 0 && (
                  <span>{results.length} result{results.length !== 1 ? 's' : ''}</span>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdvancedSearch;
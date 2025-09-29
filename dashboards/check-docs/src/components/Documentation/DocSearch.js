import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { loadSearchIndex, searchDocuments } from '../../utils/searchLoader';

const DocSearch = ({ onSearch }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [showResults, setShowResults] = useState(false);
  const [searchData, setSearchData] = useState(null);
  const [indexLoading, setIndexLoading] = useState(true);
  const searchRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    // Load the pre-built search index
    loadSearchIndex().then(data => {
      setSearchData(data);
      setIndexLoading(false);
    });
  }, []);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (searchRef.current && !searchRef.current.contains(event.target)) {
        setShowResults(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSearch = (searchQuery) => {
    setQuery(searchQuery);
    onSearch(searchQuery);

    if (searchQuery.length > 1 && searchData) {
      const searchResults = searchDocuments(searchQuery, searchData);
      setResults(searchResults);
      setShowResults(true);
    } else {
      setResults([]);
      setShowResults(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Escape') {
      setShowResults(false);
      setQuery('');
    }
  };

  const handleResultClick = (result) => {
    setShowResults(false);
    setQuery('');
    onSearch(''); // Clear search in parent
    // Navigate to the result using React Router
    navigate(result.path);
  };

  return (
    <div className="flex-1 max-w-md mx-5 relative" ref={searchRef}>
      <div className="relative flex items-center bg-white border-2 border-gray-300 focus-within:border-ubs-red transition-colors duration-200">
        <svg
          className="absolute left-3 text-gray-600 pointer-events-none"
          width="20"
          height="20"
          viewBox="0 0 20 20"
          fill="none"
        >
          <path
            d="M9 17A8 8 0 109 1a8 8 0 000 16zM19 19l-4.35-4.35"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="square"
          />
        </svg>
        <input
          type="text"
          className="w-full py-2.5 pl-10 pr-10 text-sm border-0 outline-none bg-transparent text-gray-800 placeholder-gray-600"
          placeholder={indexLoading ? "Loading search..." : "Search documentation..."}
          value={query}
          onChange={(e) => handleSearch(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => query.length > 1 && setShowResults(true)}
          disabled={indexLoading}
        />
        {query && (
          <button
            className="absolute right-3 bg-none border-0 text-gray-600 hover:text-ubs-red cursor-pointer text-lg p-0 w-5 h-5 flex items-center justify-center transition-colors duration-200"
            onClick={() => {
              setQuery('');
              setResults([]);
              setShowResults(false);
              onSearch('');
            }}
            aria-label="Clear search"
          >
            ✕
          </button>
        )}
      </div>

      {showResults && results.length > 0 && (
        <div className="absolute top-full mt-2 left-0 right-0 bg-white border-2 border-gray-300 shadow-lg max-h-96 overflow-y-auto z-50">
          <div className="p-3 bg-gray-100 border-b border-gray-300 text-xs font-semibold text-gray-600 uppercase tracking-wide">
            {results.length} result{results.length !== 1 ? 's' : ''} found
          </div>
          {results.map((result) => {
            const doc = result.document;

            return (
              <div
                key={result.ref}
                className="p-4 border-b border-gray-200 cursor-pointer hover:bg-gray-100 transition-colors duration-200"
                onClick={() => handleResultClick(doc)}
              >
                <div className="text-sm font-semibold text-gray-800 mb-1">{doc.title}</div>
                <div className="text-xs text-gray-600 mb-2">{doc.description}</div>
                <div className="flex justify-between items-center">
                  <div className="text-xs text-ubs-red font-medium">{doc.path}</div>
                  <div className="text-xs text-gray-500">
                    {Math.min(100, Math.round(result.score * 10))}% match
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {showResults && query.length > 1 && results.length === 0 && (
        <div className="absolute top-full mt-2 left-0 right-0 bg-white border-2 border-gray-300 shadow-lg z-50">
          <div className="p-6 text-center text-gray-600 text-sm">
            No results found for "{query}"
          </div>
        </div>
      )}
    </div>
  );
};

export default DocSearch;
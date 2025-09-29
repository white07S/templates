import React, { useState } from 'react';
import { Link } from 'react-router-dom';

const DocSidebar = ({ docs, isOpen, currentDoc, searchQuery }) => {
  const [expandedCategories, setExpandedCategories] = useState({
    Introduction: true,
    Fundamentals: true,
    Reference: true,
    Guides: true,
    Support: true
  });

  const groupedDocs = docs.reduce((acc, doc) => {
    if (!acc[doc.category]) {
      acc[doc.category] = [];
    }
    acc[doc.category].push(doc);
    return acc;
  }, {});

  const filteredDocs = searchQuery
    ? docs.filter(doc =>
        doc.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        doc.description.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : null;

  const toggleCategory = (category) => {
    setExpandedCategories(prev => ({
      ...prev,
      [category]: !prev[category]
    }));
  };

  const renderDocLink = (doc) => (
    <Link
      key={doc.id}
      to={doc.path}
      className={`flex items-center p-3 text-gray-800 border-l-3 transition-all duration-200 relative ${
        currentDoc?.id === doc.id
          ? 'bg-white border-red-600 font-semibold'
          : 'border-transparent hover:bg-gray-100 hover:border-gray-600'
      }`}
    >
      <span className="flex-1 text-sm">{doc.title}</span>
      {currentDoc?.id === doc.id && (
        <span className="w-1 h-full bg-red-600 absolute right-0 top-0"></span>
      )}
    </Link>
  );

  return (
    <aside className={`${isOpen ? 'w-72' : 'w-0'} min-h-full bg-gray-50 border-r-2 border-gray-300 overflow-y-auto transition-all duration-300 flex-shrink-0`}>
      <div className={`flex flex-col h-full ${!isOpen ? 'hidden' : ''}`}>
        <div className="p-6 border-b-2 border-gray-300 bg-white">
          <h2 className="text-xl font-bold text-gray-800 uppercase tracking-wide">Documentation</h2>
        </div>

        <nav className="flex-1 p-4 overflow-y-auto">
          {searchQuery && filteredDocs ? (
            <div>
              <div className="text-sm font-semibold text-gray-600 uppercase tracking-wide mb-4">
                Search Results ({filteredDocs.length})
              </div>
              <div className="space-y-1">
                {filteredDocs.map(renderDocLink)}
              </div>
            </div>
          ) : (
            Object.entries(groupedDocs).map(([category, categoryDocs]) => (
              <div key={category} className="mb-4 border border-gray-300 bg-white">
                <button
                  className="flex items-center w-full p-4 bg-white hover:bg-gray-100 transition-colors duration-200 text-left"
                  onClick={() => toggleCategory(category)}
                  aria-expanded={expandedCategories[category]}
                >
                  <span className="mr-2 text-xs text-gray-600">
                    {expandedCategories[category] ? '▼' : '▶'}
                  </span>
                  <span className="flex-1 text-sm font-semibold text-gray-800 uppercase tracking-wide">{category}</span>
                  <span className="text-xs text-gray-600 bg-gray-200 px-2 py-1">{categoryDocs.length}</span>
                </button>
                {expandedCategories[category] && (
                  <div className="border-t border-gray-300">
                    {categoryDocs.map(renderDocLink)}
                  </div>
                )}
              </div>
            ))
          )}
        </nav>

        <div className="p-4 border-t-2 border-gray-300 bg-white">
          <div className="text-xs text-gray-600 text-center">
            <span>Version 1.0.0</span>
          </div>
        </div>
      </div>
    </aside>
  );
};

export default DocSidebar;
import React, { useState, useEffect } from 'react';
import { ChevronRight, ChevronDown, Menu, X } from 'lucide-react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

const Sidebar = ({ currentPageId, onPageSelect }) => {
  const [pages, setPages] = useState([]);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [expandedItems, setExpandedItems] = useState({});

  useEffect(() => {
    fetchPages();
  }, []);

  const fetchPages = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/pages`);
      setPages(response.data);
    } catch (error) {
      console.error('Failed to fetch pages:', error);
    }
  };

  const toggleExpand = (itemId) => {
    setExpandedItems(prev => ({
      ...prev,
      [itemId]: !prev[itemId]
    }));
  };

  const renderPageItem = (page) => {
    const hasChildren = page.children && page.children.length > 0;
    const isExpanded = expandedItems[page.id];
    const isActive = currentPageId === page.id;

    return (
      <div key={page.id} className="mb-1">
        <div
          className={`
            flex items-center px-3 py-2 cursor-pointer
            border-l-4 transition-colors
            ${isActive
              ? 'bg-red-50 border-red-600 text-red-900'
              : 'border-transparent hover:bg-gray-50 hover:border-gray-300'
            }
          `}
          onClick={() => onPageSelect(page)}
        >
          {hasChildren && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                toggleExpand(page.id);
              }}
              className="mr-1"
            >
              {isExpanded ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
            </button>
          )}
          <span className="text-sm font-medium">{page.title}</span>
        </div>
        {hasChildren && isExpanded && (
          <div className="ml-4">
            {page.children.map(child => renderPageItem(child))}
          </div>
        )}
      </div>
    );
  };

  return (
    <>
      {/* Mobile toggle button */}
      <button
        className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-white border border-gray-200"
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        {isCollapsed ? <Menu className="w-5 h-5" /> : <X className="w-5 h-5" />}
      </button>

      {/* Sidebar */}
      <div
        className={`
          fixed lg:relative top-0 left-0 h-screen bg-white border-r border-gray-200
          transition-all duration-300 z-40 flex flex-col
          ${isCollapsed ? '-translate-x-full lg:translate-x-0 lg:w-16' : 'translate-x-0 w-64'}
        `}
      >
        {/* Header */}
        <div className="p-4 border-b border-gray-200">
          {!isCollapsed && (
            <h2 className="text-lg font-bold text-gray-900">Documentation</h2>
          )}
          <button
            className="hidden lg:block mt-2"
            onClick={() => setIsCollapsed(!isCollapsed)}
          >
            {isCollapsed ? (
              <ChevronRight className="w-5 h-5" />
            ) : (
              <ChevronDown className="w-5 h-5" />
            )}
          </button>
        </div>

        {/* Navigation */}
        {!isCollapsed && (
          <div className="overflow-y-auto flex-1 py-4">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-3 mb-2">
              Pages
            </h3>
            {pages.map(page => renderPageItem(page))}
          </div>
        )}
      </div>
    </>
  );
};

export default Sidebar;
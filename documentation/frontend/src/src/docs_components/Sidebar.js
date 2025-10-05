import React, { useState, useEffect } from 'react';
import { ChevronRight, ChevronDown, Menu, X } from 'lucide-react';
import { documentationAPI } from '../components/api';

const Sidebar = ({ currentPageId, onPageSelect }) => {
  const [pages, setPages] = useState([]);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [expandedItems, setExpandedItems] = useState({});

  useEffect(() => {
    fetchPages();
  }, []);

  const fetchPages = async () => {
    try {
      const data = await documentationAPI.getPages();
      setPages(data);
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

  const renderPageItem = (page, depth = 0) => {
    const hasChildren = page.children && page.children.length > 0;
    const isExpanded = expandedItems[page.id];
    const isActive = currentPageId === page.id;
    const isDirectory = hasChildren && !page.has_index;
    const paddingLeft = `${(depth * 1.25) + 0.75}rem`;

    return (
      <div key={page.id} className="mb-1">
        <div
          className={`
            flex items-center py-2 cursor-pointer
            border-l-4 transition-colors
            ${isActive
              ? 'bg-red-50 border-red-600 text-red-900'
              : 'border-transparent hover:bg-gray-50 hover:border-gray-300'
            }
          `}
          style={{ paddingLeft }}
          onClick={() => {
            // Only navigate if it's not a directory without an index
            if (!isDirectory) {
              onPageSelect(page);
            } else {
              toggleExpand(page.id);
            }
          }}
        >
          {hasChildren && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                toggleExpand(page.id);
              }}
              className="mr-1 hover:bg-gray-200 rounded p-0.5"
            >
              {isExpanded ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
            </button>
          )}
          {!hasChildren && <span className="mr-1 w-4" />}
          <span
            className={`text-sm font-medium ${isDirectory ? 'text-gray-600' : ''}`}
          >
            {page.title}
          </span>
        </div>
        {hasChildren && isExpanded && (
          <div className="border-l border-gray-200 ml-2">
            {page.children.map(child => renderPageItem(child, depth + 1))}
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
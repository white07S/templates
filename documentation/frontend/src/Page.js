import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import Sidebar from './docs_components/Sidebar';
import MDXRenderer from './docs_components/MDXRenderer';
import Footer from './docs_components/Footer';
import AdvancedSearch from './docs_components/AdvancedSearch';
import { Loader2, AlertCircle, Search } from 'lucide-react';

const API_URL = 'http://localhost:8000';

const Page = () => {
  const [currentPage, setCurrentPage] = useState(null);
  const [pageContent, setPageContent] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isSearchOpen, setIsSearchOpen] = useState(false);

  useEffect(() => {
    // Load the first page on mount
    loadInitialPage();

    // Add keyboard shortcut for search (Cmd+K or Ctrl+K)
    const handleKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsSearchOpen(true);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const loadInitialPage = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_URL}/api/pages`);
      if (response.data && response.data.length > 0) {
        handlePageSelect(response.data[0]);
      }
    } catch (error) {
      console.error('Failed to load initial page:', error);
      setError('Failed to load documentation pages');
    } finally {
      setLoading(false);
    }
  };

  const handlePageSelect = async (page, scrollToAnchor = null) => {
    try {
      setLoading(true);
      setError(null);

      // If page object has an id, use it, otherwise assume the whole object is the page
      const pageId = page.id || page;

      const response = await axios.get(`${API_URL}/api/page/${pageId}`);
      setCurrentPage(page);
      setPageContent(response.data);

      // Scroll to anchor if provided
      if (scrollToAnchor || page.anchor) {
        setTimeout(() => {
          const anchor = scrollToAnchor || page.anchor;
          const element = document.getElementById(anchor);
          if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'start' });
          } else {
            // Try to find heading with matching text
            const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
            for (const heading of headings) {
              if (heading.textContent.toLowerCase().replace(/\s+/g, '-') === anchor) {
                heading.scrollIntoView({ behavior: 'smooth', block: 'start' });
                break;
              }
            }
          }
        }, 100);
      }
    } catch (error) {
      console.error('Failed to load page:', error);
      setError('Failed to load page content');
      setPageContent(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Advanced Search Modal */}
      <AdvancedSearch
        isOpen={isSearchOpen}
        onClose={() => setIsSearchOpen(false)}
        onNavigate={handlePageSelect}
      />

      {/* Sidebar */}
      <Sidebar
        currentPageId={currentPage?.id}
        onPageSelect={handlePageSelect}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-h-screen">
        {/* Header with Search Button */}
        <header className="bg-white border-b-2 border-gray-200 px-8 py-4">
          <div className="flex justify-between items-center">
            <h1 className="text-2xl font-bold text-gray-800">Documentation</h1>
            <button
              onClick={() => setIsSearchOpen(true)}
              className="flex items-center px-6 py-3 bg-gray-200 hover:bg-gray-300 text-white font-medium transition-colors shadow-sm"
            >
              <Search className="w-5 h-5 mr-3" />
              <span className="mr-4 text-base">Search Documentation</span>
              <kbd className="px-2 py-1 text-xs bg-gray-700 border border-gray-300 rounded">⌘K</kbd>
            </button>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 overflow-auto">
          <div className="px-8 py-8">
            {loading && (
              <div className="flex items-center justify-center py-20">
                <Loader2 className="w-8 h-8 animate-spin text-red-600" />
                <span className="ml-2 text-gray-600">Loading...</span>
              </div>
            )}

            {error && (
              <div className="bg-red-50 border border-red-200 p-4 flex items-start">
                <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 mr-2 flex-shrink-0" />
                <div>
                  <h3 className="text-red-800 font-medium">Error</h3>
                  <p className="text-red-700 text-sm mt-1">{error}</p>
                </div>
              </div>
            )}

            {!loading && !error && pageContent && (
              <>
                {/* Page Title */}
                <h1 className="text-4xl font-bold text-gray-900 mb-8 pb-4 border-b border-gray-200">
                  {pageContent.title}
                </h1>

                {/* Page Content */}
                <div className="prose prose-lg max-w-none">
                  <MDXRenderer content={pageContent.content} />
                </div>
              </>
            )}

            {!loading && !error && !pageContent && (
              <div className="text-center py-20">
                <h2 className="text-2xl font-semibold text-gray-700 mb-2">
                  Welcome to Documentation
                </h2>
                <p className="text-gray-600">
                  Select a page from the sidebar to get started
                </p>
              </div>
            )}
          </div>
        </main>

        {/* Footer */}
        {currentPage && (
          <Footer
            currentPageId={currentPage.id}
            onPageSelect={handlePageSelect}
          />
        )}
      </div>
    </div>
  );
};

export default Page;
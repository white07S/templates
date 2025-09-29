import React, { useState, useEffect } from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import DocLayout from '../components/Documentation/DocLayout';
import DocSidebar from '../components/Documentation/DocSidebar';
import DocContent from '../components/Documentation/DocContent';
import DocSearch from '../components/Documentation/DocSearch';
import DocBreadcrumb from '../components/Documentation/DocBreadcrumb';
import DocNavigation from '../components/Documentation/DocNavigation';
import { docsApi } from '../services/api';

const Docs = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [currentDoc, setCurrentDoc] = useState(null);
  const [docsList, setDocsList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const location = useLocation();

  // Fetch docs list from API
  useEffect(() => {
    const fetchDocs = async () => {
      try {
        setLoading(true);
        const docs = await docsApi.getDocsList();
        setDocsList(docs);
      } catch (err) {
        console.error('Error fetching docs list:', err);
        setError(err.message || 'Failed to fetch documentation list');
        // Fallback to local list
        setDocsList([
          {
            id: 'getting-started',
            title: 'Getting Started',
            path: '/docs/getting-started',
            description: 'Quick start guide to get you up and running',
            category: 'Introduction'
          },
          {
            id: 'core-concepts',
            title: 'Core Concepts',
            path: '/docs/core-concepts',
            description: 'Fundamental concepts and architecture',
            category: 'Fundamentals'
          },
          {
            id: 'api-reference',
            title: 'API Reference',
            path: '/docs/api-reference',
            description: 'Complete API documentation and endpoints',
            category: 'Reference'
          },
          {
            id: 'examples',
            title: 'Examples',
            path: '/docs/examples',
            description: 'Practical examples and code snippets',
            category: 'Guides'
          },
          {
            id: 'math-formulas',
            title: 'Math Formulas & Code',
            path: '/docs/math-formulas',
            description: 'Mathematical formulas and enhanced code blocks',
            category: 'Advanced'
          },
          {
            id: 'troubleshooting',
            title: 'Troubleshooting',
            path: '/docs/troubleshooting',
            description: 'Common issues and their solutions',
            category: 'Support'
          }
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchDocs();
  }, []);

  useEffect(() => {
    const currentPath = location.pathname;
    const doc = docsList.find(d => d.path === currentPath);
    setCurrentDoc(doc);
  }, [location, docsList]);

  const handleSearch = (query) => {
    setSearchQuery(query);
  };

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  if (loading && docsList.length === 0) {
    return (
      <DocLayout>
        <div className="flex justify-center items-center min-h-screen">
          <div className="flex flex-col items-center gap-4">
            <div className="w-10 h-10 border-3 border-gray-300 border-t-red-600 rounded-full animate-spin"></div>
            <span className="text-gray-600">Loading documentation...</span>
          </div>
        </div>
      </DocLayout>
    );
  }

  return (
    <DocLayout>
      <div className="flex flex-col min-h-screen bg-white">
        <div className="flex items-center p-5 bg-white border-b-2 border-red-600 shadow-sm sticky top-0 z-20">
          <button
            className="flex items-center justify-center w-10 h-10 mr-5 bg-white border-2 border-gray-800 hover:bg-gray-100 hover:border-red-600 transition-colors duration-200"
            onClick={toggleSidebar}
            aria-label="Toggle sidebar"
          >
            <div className="flex flex-col gap-1">
              <span className="w-5 h-0.5 bg-gray-800 transition-colors duration-200 hover:bg-red-600"></span>
              <span className="w-5 h-0.5 bg-gray-800 transition-colors duration-200 hover:bg-red-600"></span>
              <span className="w-5 h-0.5 bg-gray-800 transition-colors duration-200 hover:bg-red-600"></span>
            </div>
          </button>
          <DocBreadcrumb currentDoc={currentDoc} />
          <DocSearch onSearch={handleSearch} />
        </div>

        <div className="flex flex-1 relative">
          <DocSidebar
            docs={docsList}
            isOpen={sidebarOpen}
            currentDoc={currentDoc}
            searchQuery={searchQuery}
          />

          <main className="flex-1 bg-white p-8 md:p-12 overflow-auto">
            <Routes>
              <Route path="/" element={
                <DocContent docId="getting-started" />
              } />
              <Route path="/getting-started" element={
                <DocContent docId="getting-started" />
              } />
              <Route path="/core-concepts" element={
                <DocContent docId="core-concepts" />
              } />
              <Route path="/api-reference" element={
                <DocContent docId="api-reference" />
              } />
              <Route path="/examples" element={
                <DocContent docId="examples" />
              } />
              <Route path="/math-formulas" element={
                <DocContent docId="math-formulas" />
              } />
              <Route path="/troubleshooting" element={
                <DocContent docId="troubleshooting" />
              } />
            </Routes>

            {currentDoc && (
              <DocNavigation
                docs={docsList}
                currentDoc={currentDoc}
              />
            )}
          </main>
        </div>
      </div>
    </DocLayout>
  );
};

export default Docs;
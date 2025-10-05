import React, { useState, useEffect } from 'react';
import GlossaryAPI from './api';
import TermList from './TermList';
import TermForm from './TermForm';
import TermDetail from './TermDetail';
import SearchBar from './SearchBar';

const GlossaryManager = ({ userId = 'user123' }) => {
  const [terms, setTerms] = useState([]);
  const [filteredTerms, setFilteredTerms] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);
  const [totalPages, setTotalPages] = useState(0);
  const [totalItems, setTotalItems] = useState(0);

  // UI state
  const [showForm, setShowForm] = useState(false);
  const [editingTerm, setEditingTerm] = useState(null);
  const [viewingTerm, setViewingTerm] = useState(null);

  // Load all terms on component mount or when page changes
  useEffect(() => {
    loadTerms();
  }, [currentPage, itemsPerPage]);

  // Filter terms when search query changes
  useEffect(() => {
    if (searchQuery) {
      searchTerms(searchQuery);
    } else {
      loadTerms();
    }
  }, [searchQuery]);

  const loadTerms = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await GlossaryAPI.getAllTerms(currentPage, itemsPerPage);
      setTerms(data.items || []);
      setFilteredTerms(data.items || []);
      setTotalPages(data.total_pages || 0);
      setTotalItems(data.total_items || 0);
    } catch (err) {
      setError('Failed to load terms. Please try again.');
      console.error('Error loading terms:', err);
    } finally {
      setLoading(false);
    }
  };

  const searchTerms = async (query) => {
    if (!query) {
      loadTerms();
      return;
    }

    setLoading(true);
    try {
      const data = await GlossaryAPI.searchTerms(query, currentPage, itemsPerPage);
      setFilteredTerms(data.items || []);
      setTotalPages(data.total_pages || 0);
      setTotalItems(data.total_items || 0);
    } catch (err) {
      setError('Failed to search terms. Please try again.');
      console.error('Error searching terms:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (query) => {
    setSearchQuery(query);
    setCurrentPage(1); // Reset to first page on new search
  };

  const handleCreate = async (termData) => {
    try {
      await GlossaryAPI.createTerm(termData, userId);
      await loadTerms();
      setShowForm(false);
      setEditingTerm(null);
    } catch (err) {
      throw err;
    }
  };

  const handleUpdate = async (termData) => {
    try {
      await GlossaryAPI.updateTerm(editingTerm.id, termData, userId);
      await loadTerms();
      setShowForm(false);
      setEditingTerm(null);
    } catch (err) {
      throw err;
    }
  };

  // Delete functionality removed from UI

  const handleEdit = (term) => {
    setEditingTerm(term);
    setShowForm(true);
    setViewingTerm(null);
  };

  const handleView = (term) => {
    setViewingTerm(term);
    setShowForm(false);
  };

  const handleFormCancel = () => {
    setShowForm(false);
    setEditingTerm(null);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-6xl mx-auto px-4">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            Glossary Management System
          </h1>
          <p className="text-gray-600">
            Manage your organization's terminology and definitions
          </p>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-4 p-4 bg-red-50 border border-red-400 text-red-700">
            {error}
            <button
              onClick={() => setError(null)}
              className="float-right font-bold"
            >
              ×
            </button>
          </div>
        )}

        {/* Controls */}
        <div className="mb-6 space-y-4">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1">
              <SearchBar onSearch={handleSearch} />
            </div>
            <button
              onClick={() => {
                setShowForm(true);
                setEditingTerm(null);
                setViewingTerm(null);
              }}
              className="px-6 py-2 bg-gray-600 text-white hover:bg-gray-700 transition-colors"
            >
              Add New Term
            </button>
          </div>

          {searchQuery && (
            <div className="text-sm text-gray-600">
              Showing results for: <strong>{searchQuery}</strong>
              {' '}({filteredTerms.length} found)
            </div>
          )}
        </div>

        {/* Main Content Area */}
        {loading ? (
          <div className="text-center py-8">
            <div className="inline-block animate-spin h-8 w-8 border-4 border-black border-t-transparent"></div>
            <p className="mt-2 text-gray-600">Loading...</p>
          </div>
        ) : (
          <>
            {/* Term Form */}
            {showForm && (
              <div className="mb-6">
                <TermForm
                  term={editingTerm}
                  onSubmit={editingTerm ? handleUpdate : handleCreate}
                  onCancel={handleFormCancel}
                  userId={userId}
                />
              </div>
            )}

            {/* Term Detail View */}
            {viewingTerm && !showForm && (
              <div className="mb-6">
                <TermDetail
                  term={viewingTerm}
                  onEdit={handleEdit}
                  onClose={() => setViewingTerm(null)}
                />
              </div>
            )}

            {/* Terms List */}
            {!showForm && !viewingTerm && (
              <>
                <TermList
                  terms={filteredTerms}
                  onEdit={handleEdit}
                  onView={handleView}
                />

                {/* Pagination Controls */}
                {totalItems > 0 && (
                  <div className="mt-6 flex justify-between items-center">
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-gray-600">
                        Items per page:
                      </span>
                      <select
                        value={itemsPerPage}
                        onChange={(e) => {
                          setItemsPerPage(Number(e.target.value));
                          setCurrentPage(1);
                        }}
                        className="px-2 py-1 border border-gray-300 bg-white text-sm"
                      >
                        <option value={5}>5</option>
                        <option value={10}>10</option>
                        <option value={20}>20</option>
                        <option value={50}>50</option>
                      </select>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                        disabled={currentPage === 1}
                        className={`px-3 py-1 border ${
                          currentPage === 1
                            ? 'bg-gray-100 text-gray-400 border-gray-200 cursor-not-allowed'
                            : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                        }`}
                      >
                        Previous
                      </button>

                      <span className="px-3 py-1 text-sm">
                        Page {currentPage} of {totalPages}
                      </span>

                      <button
                        onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                        disabled={currentPage === totalPages}
                        className={`px-3 py-1 border ${
                          currentPage === totalPages
                            ? 'bg-gray-100 text-gray-400 border-gray-200 cursor-not-allowed'
                            : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                        }`}
                      >
                        Next
                      </button>
                    </div>

                    <div className="text-sm text-gray-600">
                      Showing {((currentPage - 1) * itemsPerPage) + 1} - {Math.min(currentPage * itemsPerPage, totalItems)} of {totalItems} items
                    </div>
                  </div>
                )}
              </>
            )}
          </>
        )}

        {/* Footer Info */}
        <div className="mt-8 text-center text-sm text-gray-500">
          <p>User ID: {userId}</p>
          <p>Total Terms: {totalItems}</p>
        </div>
      </div>
    </div>
  );
};

export default GlossaryManager;
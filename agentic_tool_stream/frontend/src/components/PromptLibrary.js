import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { Plus, Search, Filter, Edit2, Copy, Trash2, X, Check } from 'lucide-react';
import PromptCard from './PromptCard';
import PromptForm from './PromptForm';

const PromptLibrary = ({ user, onSelectPrompt }) => {
  const [prompts, setPrompts] = useState([]);
  const [filteredPrompts, setFilteredPrompts] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showForm, setShowForm] = useState(false);
  const [editingPrompt, setEditingPrompt] = useState(null);
  const [searchKeywords, setSearchKeywords] = useState('');
  const [filterUserCreated, setFilterUserCreated] = useState(null);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  useEffect(() => {
    loadPrompts();
  }, []);

  useEffect(() => {
    applyFilters();
  }, [prompts, searchKeywords, filterUserCreated]);

  const loadPrompts = async () => {
    setIsLoading(true);
    try {
      const data = await api.listPrompts();
      setPrompts(data);
      setError('');
    } catch (error) {
      console.error('Failed to load prompts:', error);
      setError('Failed to load prompts');
    } finally {
      setIsLoading(false);
    }
  };

  const applyFilters = () => {
    let filtered = [...prompts];

    if (filterUserCreated !== null) {
      filtered = filtered.filter(p => 
        filterUserCreated ? p.is_owner : !p.is_owner
      );
    }

    if (searchKeywords) {
      const keywords = searchKeywords.toLowerCase().split(',').map(k => k.trim());
      filtered = filtered.filter(prompt => {
        const promptKeywords = prompt.keywords_used_for_search.map(k => k.toLowerCase());
        return keywords.some(kw => 
          promptKeywords.some(pk => pk.includes(kw)) ||
          prompt.persona.toLowerCase().includes(kw) ||
          prompt.task.toLowerCase().includes(kw)
        );
      });
    }

    setFilteredPrompts(filtered);
  };

  const handleCreatePrompt = async (promptData) => {
    try {
      await api.createPrompt(promptData);
      setSuccess('Prompt created successfully');
      setShowForm(false);
      setEditingPrompt(null);
      loadPrompts();
      setTimeout(() => setSuccess(''), 3000);
    } catch (error) {
      setError(error.message || 'Failed to create prompt');
    }
  };

  const handleUpdatePrompt = async (promptId, updateData) => {
    try {
      await api.updatePrompt(promptId, updateData);
      setSuccess('Prompt updated successfully');
      setShowForm(false);
      setEditingPrompt(null);
      loadPrompts();
      setTimeout(() => setSuccess(''), 3000);
    } catch (error) {
      setError(error.message || 'Failed to update prompt');
    }
  };

  const handleDeletePrompt = async (promptId) => {
    if (!window.confirm('Are you sure you want to delete this prompt?')) return;
    
    try {
      await api.deletePrompt(promptId);
      setSuccess('Prompt deleted successfully');
      loadPrompts();
      setTimeout(() => setSuccess(''), 3000);
    } catch (error) {
      setError(error.message || 'Failed to delete prompt');
    }
  };

  const handleCopyPrompt = async (promptId) => {
    try {
      const result = await api.copyPrompt(promptId);
      setSuccess('Prompt copied successfully');
      loadPrompts();
      setTimeout(() => setSuccess(''), 3000);
    } catch (error) {
      setError(error.message || 'Failed to copy prompt');
    }
  };

  const handleEditPrompt = (prompt) => {
    setEditingPrompt(prompt);
    setShowForm(true);
  };

  const handleSelectPrompt = (prompt) => {
    if (onSelectPrompt) {
      onSelectPrompt(prompt);
    }
  };

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Header */}
      <div className="border-b-2 border-black bg-white p-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Prompt Library</h1>
            <p className="text-sm text-gray-600 mt-1">Browse and manage AI prompts</p>
          </div>
          <button
            onClick={() => {
              setEditingPrompt(null);
              setShowForm(true);
            }}
            className="flex items-center gap-2 px-4 py-2 bg-black text-white border-2 border-black hover:bg-gray-800 transition-colors"
          >
            <Plus className="w-4 h-4" />
            Create Prompt
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="border-b border-gray-200 bg-white p-4">
        <div className="flex gap-4 items-center">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search by keywords (comma-separated)..."
              value={searchKeywords}
              onChange={(e) => setSearchKeywords(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border-2 border-black focus:outline-none focus:ring-2 focus:ring-gray-400"
            />
          </div>
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-gray-600" />
            <button
              onClick={() => setFilterUserCreated(filterUserCreated === true ? null : true)}
              className={`px-3 py-1 border-2 ${
                filterUserCreated === true 
                  ? 'border-black bg-black text-white' 
                  : 'border-gray-300 hover:border-black'
              } transition-colors`}
            >
              My Prompts
            </button>
            <button
              onClick={() => setFilterUserCreated(filterUserCreated === false ? null : false)}
              className={`px-3 py-1 border-2 ${
                filterUserCreated === false 
                  ? 'border-black bg-black text-white' 
                  : 'border-gray-300 hover:border-black'
              } transition-colors`}
            >
              Community
            </button>
          </div>
        </div>
      </div>

      {/* Notifications */}
      {error && (
        <div className="mx-4 mt-4 p-3 bg-red-100 border-2 border-red-600 text-red-800">
          {error}
          <button onClick={() => setError('')} className="float-right">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}
      {success && (
        <div className="mx-4 mt-4 p-3 bg-green-100 border-2 border-green-600 text-green-800">
          {success}
          <button onClick={() => setSuccess('')} className="float-right">
            <Check className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Prompt List */}
      <div className="flex-1 overflow-y-auto p-4">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-black border-t-transparent"></div>
              <p className="mt-2 text-gray-600">Loading prompts...</p>
            </div>
          </div>
        ) : filteredPrompts.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <p className="text-gray-600">No prompts found</p>
              <button
                onClick={() => {
                  setEditingPrompt(null);
                  setShowForm(true);
                }}
                className="mt-4 px-4 py-2 border-2 border-black hover:bg-black hover:text-white transition-colors"
              >
                Create your first prompt
              </button>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredPrompts.map(prompt => (
              <PromptCard
                key={prompt.id}
                prompt={prompt}
                onSelect={() => handleSelectPrompt(prompt)}
                onEdit={() => handleEditPrompt(prompt)}
                onCopy={() => handleCopyPrompt(prompt.id)}
                onDelete={() => handleDeletePrompt(prompt.id)}
                isOwner={prompt.is_owner}
              />
            ))}
          </div>
        )}
      </div>

      {/* Prompt Form Modal */}
      {showForm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white w-full max-w-2xl max-h-[90vh] overflow-y-auto border-2 border-black">
            <PromptForm
              prompt={editingPrompt}
              onSubmit={editingPrompt ? 
                (data) => handleUpdatePrompt(editingPrompt.id, data) : 
                handleCreatePrompt
              }
              onCancel={() => {
                setShowForm(false);
                setEditingPrompt(null);
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default PromptLibrary;
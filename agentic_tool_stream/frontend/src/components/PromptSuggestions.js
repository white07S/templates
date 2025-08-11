import React, { useState, useEffect } from 'react';
import { Sparkles, ChevronRight, X } from 'lucide-react';
import { api } from '../api';

const PromptSuggestions = ({ onSelectPrompt, onShowMore }) => {
  const [suggestions, setSuggestions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    loadSuggestions();
  }, []);

  const loadSuggestions = async () => {
    setIsLoading(true);
    try {
      const data = await api.getPromptSuggestions(4);
      setSuggestions(data);
    } catch (error) {
      console.error('Failed to load prompt suggestions:', error);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading || suggestions.length === 0) {
    return null;
  }

  return (
    <div className="px-4 py-3 border-t border-gray-200 bg-gray-50">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-gray-600" />
          <span className="text-sm font-medium text-gray-700">Suggested Prompts</span>
        </div>
        <button
          onClick={onShowMore}
          className="flex items-center gap-1 text-sm text-blue-600 hover:text-blue-700 transition-colors"
        >
          <span>View All</span>
          <ChevronRight className="w-3 h-3" />
        </button>
      </div>
      
      <div className="flex gap-2 overflow-x-auto pb-2">
        {suggestions.map((prompt) => (
          <button
            key={prompt.id}
            onClick={() => onSelectPrompt(prompt)}
            className="flex-shrink-0 px-3 py-2 bg-white border-2 border-gray-300 hover:border-black transition-colors group"
          >
            <div className="text-left max-w-[200px]">
              <p className="text-xs font-medium text-gray-900 truncate">
                {prompt.persona}
              </p>
              <p className="text-xs text-gray-600 truncate mt-1">
                {prompt.task}
              </p>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

export default PromptSuggestions;
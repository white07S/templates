import React from 'react';
import { Edit2, Copy, Trash2, User, Hash, Database, Play } from 'lucide-react';

const PromptCard = ({ prompt, onSelect, onEdit, onCopy, onDelete, isOwner }) => {
  const truncateText = (text, maxLength = 100) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  return (
    <div className="border-2 border-black bg-white hover:shadow-lg transition-shadow">
      <div className="p-4">
        <div className="flex items-start justify-between mb-2">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <User className="w-4 h-4 text-gray-600" />
              <span className="font-semibold text-sm">{prompt.persona}</span>
            </div>
            {isOwner && (
              <span className="inline-block px-2 py-1 text-xs bg-black text-white">
                Your Prompt
              </span>
            )}
          </div>
        </div>

        <div className="mb-3">
          <p className="text-gray-800 text-sm mb-2">
            <strong>Task:</strong> {truncateText(prompt.task)}
          </p>
          {prompt.if_task_need_data && (
            <div className="flex items-center gap-1 text-xs text-gray-600">
              <Database className="w-3 h-3" />
              <span>Requires data input</span>
            </div>
          )}
        </div>

        <div className="mb-3">
          <div className="flex items-center gap-1 mb-1">
            <Hash className="w-3 h-3 text-gray-600" />
            <span className="text-xs text-gray-600">Keywords:</span>
          </div>
          <div className="flex flex-wrap gap-1">
            {prompt.keywords_used_for_search.slice(0, 3).map((keyword, index) => (
              <span
                key={index}
                className="px-2 py-1 text-xs border border-gray-300 bg-gray-50"
              >
                {keyword}
              </span>
            ))}
            {prompt.keywords_used_for_search.length > 3 && (
              <span className="px-2 py-1 text-xs text-gray-500">
                +{prompt.keywords_used_for_search.length - 3} more
              </span>
            )}
          </div>
        </div>

        <div className="flex gap-2 pt-3 border-t border-gray-200">
          <button
            onClick={onSelect}
            className="flex-1 flex items-center justify-center gap-1 px-3 py-2 bg-black text-white hover:bg-gray-800 transition-colors"
          >
            <Play className="w-3 h-3" />
            <span className="text-xs">Use</span>
          </button>
          {isOwner && (
            <button
              onClick={onEdit}
              className="flex items-center justify-center p-2 border-2 border-gray-300 hover:border-black transition-colors"
              title="Edit"
            >
              <Edit2 className="w-3 h-3" />
            </button>
          )}
          <button
            onClick={onCopy}
            className="flex items-center justify-center p-2 border-2 border-gray-300 hover:border-black transition-colors"
            title="Copy"
          >
            <Copy className="w-3 h-3" />
          </button>
          {isOwner && (
            <button
              onClick={onDelete}
              className="flex items-center justify-center p-2 border-2 border-red-300 hover:border-red-600 text-red-600 transition-colors"
              title="Delete"
            >
              <Trash2 className="w-3 h-3" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default PromptCard;
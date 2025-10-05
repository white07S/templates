import React from 'react';

const TermDetail = ({ term, onEdit, onClose }) => {
  if (!term) return null;

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="bg-white border border-gray-300">
      <div className="bg-gray-600 text-white p-4">
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-bold">Term Details</h2>
          <button
            onClick={onClose}
            className="text-white hover:text-gray-300 text-2xl leading-none"
          >
            ×
          </button>
        </div>
      </div>

      <div className="p-6">
        <div className="mb-6">
          <h3 className="text-2xl font-bold mb-2 text-gray-800">{term.term}</h3>
          <div className="text-gray-500 text-sm mb-4">
            <span>ID: {term.id}</span>
          </div>
        </div>

        <div className="mb-6">
          <h4 className="text-sm font-semibold uppercase tracking-wide text-gray-600 mb-2">
            Definition
          </h4>
          <p className="text-gray-800 whitespace-pre-wrap">{term.definition}</p>
        </div>

        {term.synonyms && term.synonyms.length > 0 && (
          <div className="mb-6">
            <h4 className="text-sm font-semibold uppercase tracking-wide text-gray-600 mb-2">
              Synonyms
            </h4>
            <div className="flex flex-wrap gap-2">
              {term.synonyms.map((synonym, index) => (
                <span
                  key={index}
                  className="px-3 py-1 bg-gray-100 border border-gray-300 text-sm"
                >
                  {synonym}
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="mb-6 grid grid-cols-2 gap-4">
          <div>
            <h4 className="text-sm font-semibold uppercase tracking-wide text-gray-600 mb-1">
              Created At
            </h4>
            <p className="text-gray-800">{formatDate(term.createdAt)}</p>
          </div>
          <div>
            <h4 className="text-sm font-semibold uppercase tracking-wide text-gray-600 mb-1">
              Updated At
            </h4>
            <p className="text-gray-800">{formatDate(term.updatedAt)}</p>
          </div>
        </div>

        <div className="flex gap-3 pt-4 border-t border-gray-200">
          <button
            onClick={() => onEdit(term)}
            className="px-4 py-2 bg-white border border-gray-400 text-gray-700 hover:bg-gray-100 transition-colors"
          >
            Edit Term
          </button>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-200 text-gray-700 hover:bg-gray-300 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default TermDetail;
import React, { useState } from 'react';

const TermList = ({ terms, onEdit, onView }) => {
  const [selectedTermId, setSelectedTermId] = useState(null);

  if (!terms || terms.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No terms found. Create your first term to get started.
      </div>
    );
  }

  const handleRowClick = (termId) => {
    setSelectedTermId(selectedTermId === termId ? null : termId);
  };

  return (
    <div className="bg-white border border-gray-300">
      <table className="w-full">
        <thead className="bg-gray-600 text-white">
          <tr>
            <th className="px-4 py-3 text-left">Term</th>
            <th className="px-4 py-3 text-left">Definition</th>
            <th className="px-4 py-3 text-left">Synonyms</th>
            <th className="px-4 py-3 text-center w-24">Actions</th>
          </tr>
        </thead>
        <tbody>
          {terms.map((term, index) => (
            <tr
              key={term.id}
              className={`cursor-pointer transition-colors ${
                index % 2 === 0 ? 'bg-white hover:bg-gray-50' : 'bg-gray-50 hover:bg-gray-100'
              }`}
              onClick={() => handleRowClick(term.id)}
            >
              <td className="px-4 py-3 border-b border-gray-200 font-medium">
                {term.term}
              </td>
              <td className="px-4 py-3 border-b border-gray-200">
                {term.definition.length > 80
                  ? `${term.definition.substring(0, 80)}...`
                  : term.definition}
              </td>
              <td className="px-4 py-3 border-b border-gray-200">
                {term.synonyms && term.synonyms.length > 0 ? (
                  <span className="text-sm text-gray-600">
                    {term.synonyms.slice(0, 2).join(', ')}
                    {term.synonyms.length > 2 && ` (+${term.synonyms.length - 2} more)`}
                  </span>
                ) : (
                  <span className="text-sm text-gray-400">—</span>
                )}
              </td>
              <td className="px-4 py-3 border-b border-gray-200 relative">
                <div className="flex justify-center">
                  {selectedTermId === term.id ? (
                    <div className="flex gap-2" onClick={(e) => e.stopPropagation()}>
                      <button
                        onClick={() => {
                          onView(term);
                          setSelectedTermId(null);
                        }}
                        className="px-3 py-1 bg-gray-100 border border-gray-400 text-gray-700 hover:bg-gray-200 transition-colors"
                      >
                        View
                      </button>
                      <button
                        onClick={() => {
                          onEdit(term);
                          setSelectedTermId(null);
                        }}
                        className="px-3 py-1 bg-gray-100 border border-gray-400 text-gray-700 hover:bg-gray-200 transition-colors"
                      >
                        Edit
                      </button>
                    </div>
                  ) : (
                    <span className="text-gray-400 text-sm">Click to select</span>
                  )}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default TermList;
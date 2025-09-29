import React from 'react';
import { Link } from 'react-router-dom';

const DocNavigation = ({ docs, currentDoc }) => {
  const currentIndex = docs.findIndex(doc => doc.id === currentDoc?.id);
  const previousDoc = currentIndex > 0 ? docs[currentIndex - 1] : null;
  const nextDoc = currentIndex < docs.length - 1 ? docs[currentIndex + 1] : null;

  if (!previousDoc && !nextDoc) return null;

  return (
    <div className="mt-12 pt-6 border-t-2 border-gray-300">
      <div className="flex justify-between gap-6">
        {previousDoc ? (
          <Link to={previousDoc.path} className="flex items-center p-4 bg-white border-2 border-gray-300 hover:border-ubs-red hover:bg-gray-100 transition-all duration-200 flex-1 max-w-sm no-underline">
            <span className="text-xl text-ubs-red mr-3">←</span>
            <div className="flex flex-col">
              <span className="text-xs text-gray-600 uppercase tracking-wide mb-1">Previous</span>
              <span className="text-base font-semibold text-gray-800">{previousDoc.title}</span>
            </div>
          </Link>
        ) : (
          <div className="flex-1 max-w-sm"></div>
        )}

        {nextDoc ? (
          <Link to={nextDoc.path} className="flex items-center justify-end p-4 bg-white border-2 border-gray-300 hover:border-ubs-red hover:bg-gray-100 transition-all duration-200 flex-1 max-w-sm no-underline">
            <div className="flex flex-col text-right">
              <span className="text-xs text-gray-600 uppercase tracking-wide mb-1">Next</span>
              <span className="text-base font-semibold text-gray-800">{nextDoc.title}</span>
            </div>
            <span className="text-xl text-ubs-red ml-3">→</span>
          </Link>
        ) : (
          <div className="flex-1 max-w-sm"></div>
        )}
      </div>
    </div>
  );
};

export default DocNavigation;
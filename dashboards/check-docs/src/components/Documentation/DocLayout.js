import React from 'react';

const DocLayout = ({ children }) => {
  return (
    <div className="min-h-screen bg-white">
      <div className="w-full">
        {children}
      </div>
    </div>
  );
};

export default DocLayout;
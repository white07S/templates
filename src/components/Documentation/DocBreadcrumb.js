import React from 'react';
import { Link } from 'react-router-dom';

const DocBreadcrumb = ({ currentDoc }) => {
  const breadcrumbs = [
    { label: 'Home', path: '/' },
    { label: 'Documentation', path: '/docs' }
  ];

  if (currentDoc) {
    breadcrumbs.push({
      label: currentDoc.category,
      path: `/docs#${currentDoc.category.toLowerCase()}`
    });
    breadcrumbs.push({
      label: currentDoc.title,
      path: currentDoc.path,
      active: true
    });
  }

  return (
    <nav className="mr-auto" aria-label="Breadcrumb">
      <ol className="flex items-center list-none m-0 p-0">
        {breadcrumbs.map((item, index) => (
          <li key={index} className="flex items-center text-sm">
            {item.active ? (
              <span className="text-gray-800 font-semibold px-2 py-1" aria-current="page">
                {item.label}
              </span>
            ) : (
              <>
                <Link to={item.path} className="text-gray-600 hover:text-ubs-red px-2 py-1 transition-colors duration-200 no-underline">
                  {item.label}
                </Link>
                {index < breadcrumbs.length - 1 && (
                  <span className="text-gray-600 mx-1" aria-hidden="true">
                    /
                  </span>
                )}
              </>
            )}
          </li>
        ))}
      </ol>
    </nav>
  );
};

export default DocBreadcrumb;
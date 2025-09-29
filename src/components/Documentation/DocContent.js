import React, { useState, useEffect, Suspense } from 'react';
import { MDXProvider } from '@mdx-js/react';
import CodeBlock from './CodeBlock';
import MathFormula from './MathFormula';
import MermaidDiagram from './MermaidDiagram';

// Helper function to process inline math in text content
const processInlineMath = (children) => {
  if (typeof children === 'string') {
    // Split by dollar signs for inline math
    const parts = children.split(/\$([^$]+)\$/g);
    if (parts.length > 1) {
      return parts.map((part, index) => {
        if (index % 2 === 1) {
          // This is math content - unescape the braces
          const unescapedFormula = part.replace(/\\\{/g, '{').replace(/\\\}/g, '}');
          return <MathFormula key={index} formula={unescapedFormula} inline={true} />;
        }
        return part;
      });
    }
  } else if (Array.isArray(children)) {
    return children.map((child, index) => {
      if (typeof child === 'string') {
        return processInlineMath(child);
      }
      return child;
    });
  }
  return children;
};

// MDX component mapping
const components = {
  MathFormula,
  MermaidDiagram,
  code: ({ children, className }) => {
    const language = className ? className.replace('language-', '') : '';
    if (language) {
      return <CodeBlock code={children} language={language} />;
    }
    return (
      <code className="inline-code px-1 py-0.5 bg-gray-200 text-sm font-mono text-gray-800">
        {children}
      </code>
    );
  },
  pre: ({ children }) => {
    // Check if this is a code block with mermaid or math content
    if (children?.props?.className === 'language-mermaid') {
      return (
        <Suspense fallback={<div className="flex justify-center items-center h-64 bg-gray-50">Loading diagram...</div>}>
          <MermaidDiagram chart={children.props.children} />
        </Suspense>
      );
    }
    if (children?.props?.className === 'language-math' || children?.props?.className === 'language-latex') {
      return <MathFormula formula={children.props.children} inline={false} />;
    }
    return children;
  },
  h1: (props) => <h1 className="text-3xl font-bold mb-4" {...props}>{processInlineMath(props.children)}</h1>,
  h2: (props) => <h2 className="text-2xl font-semibold mb-3" {...props}>{processInlineMath(props.children)}</h2>,
  h3: (props) => <h3 className="text-xl font-semibold mb-2" {...props}>{processInlineMath(props.children)}</h3>,
  h4: (props) => <h4 className="text-lg font-semibold mb-2" {...props}>{processInlineMath(props.children)}</h4>,
  p: (props) => <p className="mb-4">{processInlineMath(props.children)}</p>,
  ul: (props) => <ul className="mdx-list list-disc pl-6 mb-4" {...props} />,
  ol: (props) => <ol className="mdx-list list-decimal pl-6 mb-4" {...props} />,
  li: (props) => <li className="mb-1">{processInlineMath(props.children)}</li>,
  a: (props) => <a className="mdx-link text-blue-600 hover:underline" {...props} />,
  blockquote: (props) => (
    <blockquote className="mdx-blockquote border-l-4 border-gray-300 pl-4 italic my-4" {...props} />
  ),
  table: (props) => <table className="mdx-table w-full border-collapse mb-4" {...props} />,
  thead: (props) => <thead className="bg-gray-100" {...props} />,
  th: (props) => <th className="border border-gray-300 px-4 py-2 text-left" {...props} />,
  td: (props) => <td className="border border-gray-300 px-4 py-2" {...props} />,
};

const DocContent = ({ docId }) => {
  const [MDXContent, setMDXContent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadDocument = async () => {
      setLoading(true);
      setError(null);
      setMDXContent(null);

      try {
        // Dynamically import the MDX file directly using webpack's dynamic import
        const mdxModule = await import(`../../docs/${docId}.mdx`)
          .catch(() => {
            // If direct import fails, try fetching from public folder
            return import(`../../docs/${docId}.mdx`);
          });

        if (mdxModule && mdxModule.default) {
          setMDXContent(() => mdxModule.default);
        } else {
          setError('Document not found');
        }
      } catch (err) {
        console.error('Error loading document:', err);
        setError('Failed to load document');
      } finally {
        setLoading(false);
      }
    };

    if (docId) {
      loadDocument();
    }
  }, [docId]);

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-96">
        <div className="flex flex-col items-center gap-4">
          <div className="w-10 h-10 border-3 border-gray-300 border-t-red-600 animate-spin"></div>
          <span className="text-gray-600">Loading documentation...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex justify-center items-center min-h-96">
        <div className="text-center p-6 bg-gray-100 border-2 border-red-600">
          <h2 className="text-red-600 text-xl font-bold mb-3">Error Loading Document</h2>
          <p className="text-gray-800">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <MDXProvider components={components}>
      <article className="w-full">
        <div className="bg-white border border-gray-300 p-6 md:p-8 shadow-sm prose prose-lg max-w-none">
          <Suspense fallback={<div>Loading content...</div>}>
            {MDXContent && <MDXContent />}
          </Suspense>
        </div>
      </article>
    </MDXProvider>
  );
};

export default DocContent;
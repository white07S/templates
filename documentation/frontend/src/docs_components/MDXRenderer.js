import React, { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vs } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check, ZoomIn, ZoomOut, Move } from 'lucide-react';
import mermaid from 'mermaid';
import 'katex/dist/katex.min.css';
import 'highlight.js/styles/github.css';

const API_URL = 'http://localhost:8000';

// Helper function to extract text content from React children
const extractTextFromChildren = (children) => {
  if (typeof children === 'string') {
    return children;
  }
  if (React.isValidElement(children)) {
    return extractTextFromChildren(children.props.children);
  }
  if (Array.isArray(children)) {
    return children.map(child => extractTextFromChildren(child)).join('');
  }
  if (children == null) {
    return '';
  }
  return String(children);
};

// Initialize Mermaid
mermaid.initialize({
  startOnLoad: true,
  theme: 'default',
  securityLevel: 'loose',
  themeCSS: `
    .node rect { fill: #fff; stroke: #000; stroke-width: 1px; }
    .node polygon { fill: #fff; stroke: #000; stroke-width: 1px; }
    .node circle { fill: #fff; stroke: #000; stroke-width: 1px; }
  `
});

const CodeBlock = ({ language, children }) => {
  const [copied, setCopied] = useState(false);
  const codeString = extractTextFromChildren(children);

  const handleCopy = () => {
    navigator.clipboard.writeText(codeString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative my-4 border border-gray-200">
      <div className="flex justify-between items-center px-4 py-2 bg-gray-50 border-b border-gray-200">
        <button
          onClick={handleCopy}
          className="p-1 hover:bg-gray-200 transition-colors"
          title="Copy to clipboard"
        >
          {copied ? (
            <Check className="w-4 h-4 text-green-600" />
          ) : (
            <Copy className="w-4 h-4 text-gray-600" />
          )}
        </button>
        <span className="text-xs text-gray-600 font-mono">{language || 'text'}</span>
      </div>
      <div className="overflow-x-auto">
        <SyntaxHighlighter
          language={language || 'text'}
          style={vs}
          customStyle={{
            margin: 0,
            padding: '1rem',
            background: '#fff',
            fontSize: '0.875rem',
          }}
          codeTagProps={{
            style: {
              fontFamily: 'Consolas, Monaco, "Andale Mono", "Ubuntu Mono", monospace',
            }
          }}
        >
          {codeString}
        </SyntaxHighlighter>
      </div>
    </div>
  );
};

const MermaidDiagram = ({ children }) => {
  const [svg, setSvg] = useState('');
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef(null);

  useEffect(() => {
    const renderDiagram = async () => {
      try {
        // Clean the children string
        const diagramDefinition = extractTextFromChildren(children).trim();

        // Generate a valid ID (no dots allowed)
        const uniqueId = `mermaid-${Date.now()}-${Math.floor(Math.random() * 100000)}`;

        // Initialize mermaid if needed
        mermaid.initialize({
          startOnLoad: false,
          theme: 'default',
          securityLevel: 'loose',
        });

        // Render the diagram
        const { svg: renderedSvg } = await mermaid.render(uniqueId, diagramDefinition);
        setSvg(renderedSvg);
      } catch (error) {
        console.error('Mermaid rendering error:', error);
        // Try to show the error message
        setSvg(`<div style="padding: 20px; color: red;">Error rendering Mermaid diagram: ${error.message}</div>`);
      }
    };
    renderDiagram();
  }, [children]);

  const handleZoomIn = () => setScale(prev => Math.min(prev + 0.1, 3));
  const handleZoomOut = () => setScale(prev => Math.max(prev - 0.1, 0.5));
  const handleReset = () => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
  };

  const handleMouseDown = (e) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      setPosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      });
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, dragStart]);

  return (
    <div className="my-6 border border-gray-200">
      <div className="flex gap-2 p-2 bg-gray-50 border-b border-gray-200">
        <button
          onClick={handleZoomIn}
          className="p-1 hover:bg-gray-200 transition-colors"
          title="Zoom in"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          onClick={handleZoomOut}
          className="p-1 hover:bg-gray-200 transition-colors"
          title="Zoom out"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button
          onClick={handleReset}
          className="p-1 hover:bg-gray-200 transition-colors"
          title="Reset view"
        >
          <Move className="w-4 h-4" />
        </button>
      </div>
      <div
        ref={containerRef}
        className="overflow-hidden relative bg-white"
        style={{ height: '400px', cursor: isDragging ? 'grabbing' : 'grab' }}
        onMouseDown={handleMouseDown}
      >
        <div
          style={{
            transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
            transformOrigin: 'center',
            transition: isDragging ? 'none' : 'transform 0.2s',
          }}
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      </div>
    </div>
  );
};

const BlockQuote = ({ children, className }) => {
  // Get text content from children safely
  const getTextContent = (node) => {
    if (typeof node === 'string') return node;
    if (React.isValidElement(node)) {
      if (node.props.children) {
        if (Array.isArray(node.props.children)) {
          return node.props.children.map(child => getTextContent(child)).join('');
        }
        return getTextContent(node.props.children);
      }
    }
    if (Array.isArray(node)) {
      return node.map(child => getTextContent(child)).join('');
    }
    return '';
  };

  const content = getTextContent(children);
  let blockType = 'default';
  let title = '';

  if (content.includes('Warning:') || content.includes('Caution:')) {
    blockType = 'warning';
    title = 'Warning';
  } else if (content.includes('Success:')) {
    blockType = 'success';
    title = 'Success';
  } else if (content.includes('Note:') || content.includes('Important:')) {
    blockType = 'info';
    title = 'Note';
  }

  const blockStyles = {
    warning: 'border-yellow-500 bg-yellow-50 text-yellow-900',
    success: 'border-green-500 bg-green-50 text-green-900',
    info: 'border-blue-500 bg-blue-50 text-blue-900',
    default: 'border-gray-300 bg-gray-50 text-gray-900'
  };

  return (
    <blockquote className={`border-l-4 p-4 my-4 ${blockStyles[blockType]}`}>
      {title && <strong className="block mb-2">{title}</strong>}
      {children}
    </blockquote>
  );
};

const MDXRenderer = ({ content }) => {
  const components = {
    code({ node, inline, className, children, ...props }) {
      const match = /language-(\w+)/.exec(className || '');
      const language = match ? match[1] : '';

      // Check if it's a Mermaid diagram
      if (language === 'mermaid') {
        return <MermaidDiagram>{extractTextFromChildren(children).replace(/\n$/, '')}</MermaidDiagram>;
      }

      // Inline code
      if (inline) {
        return (
          <code className="px-1 py-0.5 bg-gray-100 text-red-600 font-mono text-sm" {...props}>
            {extractTextFromChildren(children)}
          </code>
        );
      }

      // Code block
      return <CodeBlock language={language}>{extractTextFromChildren(children).replace(/\n$/, '')}</CodeBlock>;
    },
    blockquote: BlockQuote,
    img({ src, alt }) {
      // Remove 'images/' prefix if it exists since we'll add it in the API path
      const imagePath = src.replace(/^images\//, '');
      const fullSrc = src.startsWith('http') ? src : `${API_URL}/api/image/${imagePath}`;
      return (
        <figure className="my-6">
          <img
            src={fullSrc}
            alt={alt}
            className="mx-auto max-w-full h-auto border border-gray-200"
          />
          {alt && (
            <figcaption className="text-center text-sm text-gray-600 mt-2 italic">
              {alt}
            </figcaption>
          )}
        </figure>
      );
    },
    table({ children }) {
      return (
        <div className="overflow-x-auto my-4">
          <table className="min-w-full border border-gray-200">
            {children}
          </table>
        </div>
      );
    },
    thead({ children }) {
      return <thead className="bg-gray-50 border-b border-gray-200">{children}</thead>;
    },
    th({ children }) {
      return (
        <th className="px-4 py-2 text-left text-xs font-medium text-gray-700 uppercase tracking-wider border-r border-gray-200 last:border-r-0">
          {children}
        </th>
      );
    },
    td({ children }) {
      return (
        <td className="px-4 py-2 text-sm text-gray-900 border-r border-gray-200 last:border-r-0">
          {children}
        </td>
      );
    },
    tr({ children }) {
      return <tr className="border-b border-gray-200 last:border-b-0">{children}</tr>;
    },
    h1({ children }) {
      const id = extractTextFromChildren(children).toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]/g, '');
      return <h1 id={id} className="text-3xl font-bold text-gray-900 mt-8 mb-4">{children}</h1>;
    },
    h2({ children }) {
      const id = extractTextFromChildren(children).toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]/g, '');
      return <h2 id={id} className="text-2xl font-bold text-gray-900 mt-6 mb-3">{children}</h2>;
    },
    h3({ children }) {
      const id = extractTextFromChildren(children).toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]/g, '');
      return <h3 id={id} className="text-xl font-bold text-gray-900 mt-4 mb-2">{children}</h3>;
    },
    p({ children }) {
      return <p className="text-gray-700 leading-relaxed mb-4">{children}</p>;
    },
    ul({ children }) {
      return <ul className="list-disc list-inside ml-4 mb-4 text-gray-700">{children}</ul>;
    },
    ol({ children }) {
      return <ol className="list-decimal list-inside ml-4 mb-4 text-gray-700">{children}</ol>;
    },
    li({ children }) {
      return <li className="mb-1">{children}</li>;
    },
    a({ href, children }) {
      return (
        <a
          href={href}
          className="text-red-600 hover:text-red-800 underline"
          target={href.startsWith('http') ? '_blank' : undefined}
          rel={href.startsWith('http') ? 'noopener noreferrer' : undefined}
        >
          {children}
        </a>
      );
    },
  };

  return (
    <div className="prose prose-lg max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default MDXRenderer;
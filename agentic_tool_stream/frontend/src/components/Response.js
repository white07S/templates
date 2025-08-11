import React, { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

// Helper function to process incomplete markdown during streaming
const processStreamingMarkdown = (content, isStreaming = false) => {
  if (!isStreaming) return content;
  
  let processed = content;
  
  // Handle incomplete math blocks - improved logic
  // Check for unclosed block math ($$)
  const blockMathParts = processed.split('$$');
  if (blockMathParts.length % 2 === 0) {
    // Odd number of $$, meaning one is unclosed
    processed = processed + '$$';
  }
  
  // Check for unclosed display math (\[ ... \])
  const displayMathStart = (processed.match(/\\\[/g) || []).length;
  const displayMathEnd = (processed.match(/\\\]/g) || []).length;
  if (displayMathStart > displayMathEnd) {
    processed = processed + '\\]';
  }
  
  // Check for unclosed inline math ($) - only if not inside block math
  if (!processed.endsWith('$$')) {
    // Find the last occurrence of $ that isn't part of $$
    const lastIndex = processed.lastIndexOf('$');
    if (lastIndex !== -1 && lastIndex !== processed.length - 1) {
      // Check if this $ is part of $$
      const isBlockMath = lastIndex > 0 && processed[lastIndex - 1] === '$';
      if (!isBlockMath) {
        // Count single $ (not $$) in the content
        const singleDollarCount = (processed.match(/(?<!\$)\$(?!\$)/g) || []).length;
        if (singleDollarCount % 2 !== 0) {
          processed = processed + '$';
        }
      }
    } else if (lastIndex === processed.length - 1) {
      // String ends with single $, check if we need to close it
      const beforeLast = processed.slice(0, -1);
      const singleDollarCount = (beforeLast.match(/(?<!\$)\$(?!\$)/g) || []).length;
      if (singleDollarCount % 2 === 0) {
        // Even number before, so this $ starts a new inline math
        processed = processed + '$';
      }
    }
  }
  
  // Check for unclosed inline LaTeX (\( ... \))
  const inlineMathStart = (processed.match(/\\\(/g) || []).length;
  const inlineMathEnd = (processed.match(/\\\)/g) || []).length;
  if (inlineMathStart > inlineMathEnd) {
    processed = processed + '\\)';
  }
  
  // Handle incomplete code blocks
  const codeBlockCount = (processed.match(/```/g) || []).length;
  if (codeBlockCount % 2 !== 0) {
    processed = processed + '\n```';
  }
  
  // Handle incomplete bold/italic
  const asteriskGroups = processed.match(/\*+/g) || [];
  let totalAsterisks = 0;
  asteriskGroups.forEach(group => {
    totalAsterisks += group.length;
  });
  if (totalAsterisks % 2 !== 0) {
    processed = processed + '*';
  }
  
  return processed;
};

const Response = ({ children, className = '', isStreaming = false, options = {} }) => {
  const processedContent = useMemo(() => 
    processStreamingMarkdown(children, isStreaming),
    [children, isStreaming]
  );

  return (
    <div className={`prose prose-sm max-w-none ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          p: ({ children }) => <p className="mb-3 text-gray-900">{children}</p>,
          ul: ({ children }) => <ul className="mb-3 ml-4 list-disc">{children}</ul>,
          ol: ({ children }) => <ol className="mb-3 ml-4 list-decimal">{children}</ol>,
          li: ({ children }) => <li className="mb-1">{children}</li>,
          h1: ({ children }) => <h1 className="text-2xl font-bold mb-4 text-gray-900">{children}</h1>,
          h2: ({ children }) => <h2 className="text-xl font-bold mb-3 text-gray-900">{children}</h2>,
          h3: ({ children }) => <h3 className="text-lg font-bold mb-2 text-gray-900">{children}</h3>,
          h4: ({ children }) => <h4 className="text-base font-bold mb-2 text-gray-900">{children}</h4>,
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-gray-300 pl-4 my-4 italic text-gray-700">
              {children}
            </blockquote>
          ),
          code: ({ inline, className, children }) => {
            const match = /language-(\w+)/.exec(className || '');
            const language = match ? match[1] : '';
            
            return inline ? (
              <code className="px-1 py-0.5 bg-gray-100 border border-gray-200 text-sm rounded font-mono">
                {children}
              </code>
            ) : (
              <div className="relative my-3">
                {language && (
                  <div className="absolute top-0 right-0 px-2 py-1 text-xs text-gray-600 bg-gray-200 border-b border-l border-gray-300 rounded-bl">
                    {language}
                  </div>
                )}
                <pre className="p-3 bg-gray-100 border-2 border-black overflow-x-auto rounded">
                  <code className="text-sm font-mono">{children}</code>
                </pre>
              </div>
            );
          },
          pre: ({ children }) => <>{children}</>,
          a: ({ href, children }) => (
            <a 
              href={href} 
              target="_blank" 
              rel="noopener noreferrer" 
              className="text-red-600 hover:text-red-700 underline"
            >
              {children}
            </a>
          ),
          table: ({ children }) => (
            <div className="my-4 overflow-x-auto">
              <table className="min-w-full border-collapse border-2 border-black">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-gray-100 border-b-2 border-black">
              {children}
            </thead>
          ),
          tbody: ({ children }) => (
            <tbody className="divide-y divide-gray-300">
              {children}
            </tbody>
          ),
          tr: ({ children }) => (
            <tr className="hover:bg-gray-50">
              {children}
            </tr>
          ),
          th: ({ children, style }) => {
            const align = style?.textAlign || 'left';
            return (
              <th className={`px-4 py-2 font-semibold text-gray-900 border-r border-gray-300 last:border-r-0 text-${align}`}>
                {children}
              </th>
            );
          },
          td: ({ children, style }) => {
            const align = style?.textAlign || 'left';
            return (
              <td className={`px-4 py-2 text-gray-700 border-r border-gray-300 last:border-r-0 text-${align}`}>
                {children}
              </td>
            );
          },
          hr: () => <hr className="my-4 border-t-2 border-gray-300" />,
          strong: ({ children }) => <strong className="font-bold text-gray-900">{children}</strong>,
          em: ({ children }) => <em className="italic">{children}</em>,
          del: ({ children }) => <del className="line-through text-gray-500">{children}</del>,
          input: ({ type, checked, disabled }) => {
            if (type === 'checkbox') {
              return (
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={disabled}
                  readOnly
                  className="mr-2"
                />
              );
            }
            return null;
          },
          ...options
        }}
      >
        {processedContent}
      </ReactMarkdown>
    </div>
  );
};

export default Response;
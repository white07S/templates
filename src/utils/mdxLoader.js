import React from 'react';
import CodeBlock from '../components/Documentation/CodeBlock';
import MathFormula from '../components/Documentation/MathFormula';
// Dynamic import to avoid circular dependency
const MermaidDiagram = React.lazy(() => import('../components/Documentation/MermaidDiagram'));

export const parseMDX = (content) => {
  // Remove frontmatter
  const contentWithoutFrontmatter = content.replace(/^---[\s\S]*?---\n/, '');

  // Parse the MDX content into React components
  const lines = contentWithoutFrontmatter.split('\n');
  const elements = [];
  let currentList = [];
  let inCodeBlock = false;
  let codeLanguage = '';
  let codeLines = [];
  let inTable = false;
  let tableRows = [];

  const processInlineCode = (text) => {
    if (React.isValidElement(text)) return text;
    if (typeof text !== 'string') return text;

    // First handle inline math: $...$
    const mathParts = text.split(/\$([^$]+)\$/g);
    if (mathParts.length > 1) {
      return mathParts.map((part, index) => {
        if (index % 2 === 1) {
          return <MathFormula key={index} formula={part} inline={true} />;
        }
        // Process code blocks in the remaining text
        const codeParts = part.split(/`([^`]+)`/);
        return codeParts.map((codePart, codeIndex) => {
          if (codeIndex % 2 === 1) {
            return <code key={`${index}-${codeIndex}`} className="inline-code px-1 py-0.5 bg-gray-200 text-sm font-mono text-gray-800">{codePart}</code>;
          }
          return codePart;
        });
      });
    }

    // Handle regular inline code: `...`
    const parts = text.split(/`([^`]+)`/);
    return parts.map((part, index) => {
      if (index % 2 === 1) {
        return <code key={index} className="inline-code px-1 py-0.5 bg-gray-200 text-sm font-mono text-gray-800">{part}</code>;
      }
      return part;
    });
  };

  const processStrong = (text) => {
    if (React.isValidElement(text)) return text;
    if (typeof text !== 'string') return text;

    const parts = text.split(/\*\*([^*]+)\*\*/);
    return parts.map((part, index) => {
      if (index % 2 === 1) {
        return <strong key={index}>{part}</strong>;
      }
      return processInlineCode(part);
    });
  };

  const processLinks = (text) => {
    if (React.isValidElement(text)) return text;
    if (typeof text !== 'string') return text;

    const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = linkRegex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        parts.push(processStrong(text.slice(lastIndex, match.index)));
      }
      parts.push(
        <a key={match.index} href={match[2]} className="mdx-link">
          {match[1]}
        </a>
      );
      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < text.length) {
      parts.push(processStrong(text.slice(lastIndex)));
    }

    return parts.length > 0 ? parts : processStrong(text);
  };

  const renderCodeBlock = (code, language) => {
    return <CodeBlock code={code} language={language} />;
  };

  const flushList = () => {
    if (currentList.length > 0) {
      elements.push(
        <ul key={`list-${elements.length}`} className="mdx-list">
          {currentList.map((item, index) => (
            <li key={index}>{processLinks(item)}</li>
          ))}
        </ul>
      );
      currentList = [];
    }
  };

  const parseTableRow = (row) => {
    return row.split('|').map(cell => cell.trim()).filter(cell => cell);
  };

  lines.forEach((line, index) => {
    // Handle code blocks
    if (line.startsWith('```')) {
      if (!inCodeBlock) {
        inCodeBlock = true;
        codeLanguage = line.slice(3).trim();
        codeLines = [];
      } else {
        // Check if it's a Mermaid diagram
        if (codeLanguage === 'mermaid') {
          elements.push(
            <React.Suspense key={index} fallback={<div className="flex justify-center items-center h-64 bg-gray-50">Loading diagram...</div>}>
              <MermaidDiagram chart={codeLines.join('\n')} />
            </React.Suspense>
          );
        } else if (codeLanguage === 'math' || codeLanguage === 'latex') {
          // Handle math blocks
          elements.push(
            <MathFormula key={index} formula={codeLines.join('\n')} inline={false} />
          );
        } else {
          elements.push(
            <div key={index} className="my-6 bg-gray-100 border-2 border-gray-300 overflow-hidden">
              {renderCodeBlock(codeLines.join('\n'), codeLanguage)}
            </div>
          );
        }
        inCodeBlock = false;
      }
      return;
    }

    if (inCodeBlock) {
      codeLines.push(line);
      return;
    }

    // Handle tables
    if (line.includes('|') && !inTable) {
      flushList();
      inTable = true;
      tableRows = [parseTableRow(line)];
      return;
    }

    if (inTable) {
      if (line.includes('|')) {
        if (!line.match(/^[\s|:-]+$/)) {
          tableRows.push(parseTableRow(line));
        }
      } else {
        // End of table
        if (tableRows.length > 0) {
          const headers = tableRows[0];
          const rows = tableRows.slice(1);

          elements.push(
            <table key={index} className="mdx-table">
              <thead>
                <tr>
                  {headers.map((header, i) => (
                    <th key={i}>{processLinks(header)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    {row.map((cell, cellIndex) => (
                      <td key={cellIndex}>{processLinks(cell)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          );
        }
        inTable = false;
        tableRows = [];
      }
      return;
    }

    // Handle headings
    if (line.startsWith('# ')) {
      flushList();
      elements.push(<h1 key={index}>{processLinks(line.slice(2))}</h1>);
    } else if (line.startsWith('## ')) {
      flushList();
      elements.push(<h2 key={index}>{processLinks(line.slice(3))}</h2>);
    } else if (line.startsWith('### ')) {
      flushList();
      elements.push(<h3 key={index}>{processLinks(line.slice(4))}</h3>);
    } else if (line.startsWith('#### ')) {
      flushList();
      elements.push(<h4 key={index}>{processLinks(line.slice(5))}</h4>);
    }
    // Handle lists
    else if (line.match(/^[\s]*[-*]\s+/)) {
      const listItem = line.replace(/^[\s]*[-*]\s+/, '');
      currentList.push(listItem);
    } else if (line.match(/^[\s]*\d+\.\s+/)) {
      const listItem = line.replace(/^[\s]*\d+\.\s+/, '');
      currentList.push(listItem);
    }
    // Handle blockquotes
    else if (line.startsWith('>')) {
      flushList();
      elements.push(
        <blockquote key={index} className="mdx-blockquote">
          {processLinks(line.slice(1).trim())}
        </blockquote>
      );
    }
    // Handle paragraphs
    else if (line.trim()) {
      flushList();
      elements.push(<p key={index}>{processLinks(line)}</p>);
    }
  });

  // Flush any remaining list items
  flushList();

  return elements;
};
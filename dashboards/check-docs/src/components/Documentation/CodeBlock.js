import React, { useState } from 'react';
import { Highlight, themes } from 'prism-react-renderer';

const CodeBlock = ({ code, language }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code.trim());
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy code:', err);
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = code.trim();
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      try {
        document.execCommand('copy');
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch (fallbackErr) {
        console.error('Fallback copy failed:', fallbackErr);
      }
      document.body.removeChild(textArea);
    }
  };

  if (!language || language === '') {
    return (
      <div className="relative">
        <pre className="p-4 bg-gray-100 border-0 font-mono text-sm leading-relaxed overflow-x-auto text-gray-800">
          <code>{code.trim()}</code>
        </pre>
        <button
          onClick={handleCopy}
          className={`absolute top-3 right-3 px-3 py-1 text-xs font-medium transition-all duration-200 ${
            copied
              ? 'bg-green-600 text-white'
              : 'bg-gray-200 hover:bg-gray-300 text-gray-700'
          }`}
          title={copied ? 'Copied!' : 'Copy code'}
        >
          {copied ? (
            <span className="flex items-center gap-1">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/>
              </svg>
              Copied
            </span>
          ) : (
            <span className="flex items-center gap-1">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                <path d="M16 1H4C2.9 1 2 1.9 2 3v14h2V3h12V1zm3 4H8C6.9 5 6 5.9 6 7v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>
              </svg>
              Copy
            </span>
          )}
        </button>
      </div>
    );
  }

  return (
    <div className="relative">
      <Highlight theme={themes.github} code={code.trim()} language={language}>
        {({ className, style, tokens, getLineProps, getTokenProps }) => (
          <pre
            className={`p-4 bg-gray-100 border-0 font-mono text-sm leading-relaxed overflow-x-auto ${className}`}
            style={{...style, backgroundColor: '#f5f5f5', margin: 0, color: '#1f2937'}}
          >
            {tokens.map((line, i) => (
              <div key={i} {...getLineProps({ line })}>
                {line.map((token, key) => (
                  <span key={key} {...getTokenProps({ token })} />
                ))}
              </div>
            ))}
          </pre>
        )}
      </Highlight>
      <button
        onClick={handleCopy}
        className={`absolute top-3 right-3 px-3 py-1 text-xs font-medium transition-all duration-200 border ${
          copied
            ? 'bg-green-600 text-white border-green-600'
            : 'bg-white hover:bg-gray-50 text-gray-800 border-gray-400 hover:border-gray-500'
        }`}
        title={copied ? 'Copied!' : 'Copy code'}
      >
        {copied ? (
          <span className="flex items-center gap-1">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
              <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/>
            </svg>
            Copied
          </span>
        ) : (
          <span className="flex items-center gap-1">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
              <path d="M16 1H4C2.9 1 2 1.9 2 3v14h2V3h12V1zm3 4H8C6.9 5 6 5.9 6 7v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>
            </svg>
            Copy
          </span>
        )}
      </button>
      {language && (
        <div className="absolute -top-6 left-0 px-2 py-1 text-xs bg-gray-700 text-white font-medium rounded-t-md">
          {language}
        </div>
      )}
    </div>
  );
};

export default CodeBlock;
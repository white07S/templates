import React, { useState } from 'react';
import { Copy, Check, User, Bot } from 'lucide-react';
import { motion } from 'framer-motion';
import Response from './Response';

const Message = ({ message, isUser }) => {
  const [copied, setCopied] = useState(false);
  const [showCopyButton, setShowCopyButton] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content || message);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const content = typeof message === 'string' ? message : message.content;
  const isStreaming = typeof message === 'object' && message.isStreaming;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex gap-4 p-6 ${isUser ? 'bg-gray-50' : 'bg-white'}`}
      onMouseEnter={() => setShowCopyButton(true)}
      onMouseLeave={() => setShowCopyButton(false)}
    >
      <div className={`flex-shrink-0 w-8 h-8 flex items-center justify-center border-2 border-black ${
        isUser ? 'bg-red-600' : 'bg-white'
      }`}>
        {isUser ? (
          <User size={16} className="text-white" />
        ) : (
          <Bot size={16} className="text-black" />
        )}
      </div>

      <div className="flex-1 min-w-0">
        {isUser ? (
          <p className="text-gray-900">{content}</p>
        ) : (
          <Response isStreaming={isStreaming}>{content}</Response>
        )}

        {showCopyButton && (
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={handleCopy}
            className="mt-2 flex items-center gap-1 px-2 py-1 text-xs text-gray-600 hover:text-black border border-gray-300 hover:border-black transition-colors"
          >
            {copied ? (
              <>
                <Check size={14} />
                <span>Copied</span>
              </>
            ) : (
              <>
                <Copy size={14} />
                <span>Copy</span>
              </>
            )}
          </motion.button>
        )}
      </div>
    </motion.div>
  );
};

export default Message;
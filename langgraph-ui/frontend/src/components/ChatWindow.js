import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Bot, User, Loader2, Zap, Clock } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { useChat } from '../contexts/ChatContext';
import { useAuth } from '../contexts/AuthContext';

const ChatWindow = () => {
  const [inputMessage, setInputMessage] = useState('');
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const { messages, sendMessage, isStreaming } = useChat();
  const { user } = useAuth();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || isStreaming) return;

    const message = inputMessage.trim();
    setInputMessage('');
    await sendMessage(message);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const getInitials = (name) => {
    return name
      .split(' ')
      .map(word => word[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  const getAvatarColor = (name) => {
    const colors = [
      'from-red-400 to-pink-500',
      'from-blue-400 to-indigo-500',
      'from-green-400 to-teal-500',
      'from-yellow-400 to-orange-500',
      'from-purple-400 to-pink-500',
      'from-indigo-400 to-purple-500'
    ];
    const index = name.length % colors.length;
    return colors[index];
  };

  const MessageBubble = ({ message, isUser, timestamp, tools_used, isStreaming }) => {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: isStreaming ? 0 : 0.3 }}
        className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-6`}
      >
        <div className={`flex max-w-[80%] ${isUser ? 'flex-row-reverse' : 'flex-row'} space-x-3`}>
          {/* Avatar */}
          <div className="flex-shrink-0">
            {isUser ? (
              <div className={`w-8 h-8 bg-gradient-to-br ${getAvatarColor(user?.name)} rounded-full flex items-center justify-center`}>
                <span className="text-white text-sm font-medium">
                  {getInitials(user?.name || 'U')}
                </span>
              </div>
            ) : (
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                <Bot className="w-5 h-5 text-white" />
              </div>
            )}
          </div>

          {/* Message Content */}
          <div className={`space-x-0 ${isUser ? 'mr-3' : 'ml-3'}`}>
            {/* Message Bubble */}
            <div
              className={`px-4 py-3 rounded-2xl ${
                isUser
                  ? 'bg-gradient-to-br from-blue-500 to-purple-600 text-white'
                  : 'bg-slate-700/70 text-slate-100 border border-slate-600/50'
              }`}
            >
              {isUser ? (
                <p className="text-sm leading-relaxed whitespace-pre-wrap">{message}</p>
              ) : (
                <div className="prose prose-invert prose-sm max-w-none">
                  <ReactMarkdown
                    components={{
                      p: ({ children }) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
                      code: ({ inline, children }) =>
                        inline ? (
                          <code className="bg-slate-800/50 px-1.5 py-0.5 rounded text-sm">{children}</code>
                        ) : (
                          <pre className="bg-slate-800/50 p-3 rounded-lg overflow-x-auto">
                            <code>{children}</code>
                          </pre>
                        ),
                      ul: ({ children }) => <ul className="list-disc list-inside space-y-1">{children}</ul>,
                      ol: ({ children }) => <ol className="list-decimal list-inside space-y-1">{children}</ol>,
                      li: ({ children }) => <li className="leading-relaxed">{children}</li>,
                    }}
                  >
                    {message || ''}
                  </ReactMarkdown>
                </div>
              )}
            </div>

            {/* Tools Used */}
            {!isUser && tools_used && tools_used.length > 0 && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2 }}
                className="flex items-center space-x-2 mt-2"
              >
                <Zap className="w-3 h-3 text-yellow-400" />
                <div className="flex flex-wrap gap-1">
                  {tools_used.map((tool, index) => (
                    <span
                      key={index}
                      className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-yellow-500/10 text-yellow-400 border border-yellow-500/20"
                    >
                      {tool}
                    </span>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Timestamp */}
            <div className={`flex items-center space-x-1 mt-1 ${isUser ? 'justify-end' : 'justify-start'}`}>
              <Clock className="w-3 h-3 text-slate-500" />
              <span className="text-xs text-slate-500">{formatTimestamp(timestamp)}</span>
            </div>
          </div>
        </div>
      </motion.div>
    );
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
        <AnimatePresence>
          {messages.map((msg, index) => (
            <MessageBubble
              key={`${index}-${msg.isUser ? 'user' : 'bot'}`}
              message={msg.isUser ? msg.message : msg.response}
              isUser={msg.isUser}
              timestamp={msg.timestamp}
              tools_used={msg.tools_used}
              isStreaming={!msg.isUser && isStreaming && index === messages.length - 1}
            />
          ))}
        </AnimatePresence>

        {/* Typing Indicator */}
        {isStreaming && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex justify-start"
          >
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div className="bg-slate-700/70 px-4 py-3 rounded-2xl border border-slate-600/50">
                <div className="flex items-center space-x-2">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-pulse"></div>
                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                  </div>
                  <span className="text-sm text-slate-400">AI is thinking...</span>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-slate-700/50 p-4 bg-slate-800/50 backdrop-blur-sm">
        <form onSubmit={handleSubmit} className="flex items-end space-x-3">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={isStreaming ? "AI is responding..." : "Type your message..."}
              disabled={isStreaming}
              rows={1}
              className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-2xl text-white placeholder-slate-400 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 max-h-32 min-h-[48px]"
              style={{
                height: 'auto',
                minHeight: '48px'
              }}
              onInput={(e) => {
                e.target.style.height = 'auto';
                e.target.style.height = Math.min(e.target.scrollHeight, 128) + 'px';
              }}
            />
          </div>
          
          <motion.button
            type="submit"
            disabled={!inputMessage.trim() || isStreaming}
            className="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-2xl flex items-center justify-center transition-all duration-200"
            whileHover={{ scale: !inputMessage.trim() || isStreaming ? 1 : 1.05 }}
            whileTap={{ scale: !inputMessage.trim() || isStreaming ? 1 : 0.95 }}
          >
            {isStreaming ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </motion.button>
        </form>
        
        <div className="flex items-center justify-center mt-2">
          <p className="text-xs text-slate-500">
            Press Enter to send, Shift+Enter for new line
          </p>
        </div>
      </div>
    </div>
  );
};

export default ChatWindow;
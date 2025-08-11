import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { AnimatePresence } from 'framer-motion';
import { api } from './api';
import Sidebar from './components/Sidebar';
import Message from './components/Message';
import ChatInput from './components/ChatInput';
import LoadingIndicator from './components/LoadingIndicator';
import PromptSuggestions from './components/PromptSuggestions';
import PromptPreview from './components/PromptPreview';
import PromptLibrary from './components/PromptLibrary';
import { BookOpen } from 'lucide-react';

const Chat = ({ user }) => {
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [toolCallStatus, setToolCallStatus] = useState('');
  const [showPromptLibrary, setShowPromptLibrary] = useState(false);
  const [selectedPrompt, setSelectedPrompt] = useState(null);
  const [showPromptPreview, setShowPromptPreview] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const messagesEndRef = useRef(null);

  const getTimeBasedGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 17) return 'Good afternoon';
    return 'Good evening';
  };

  const generateSessionId = () => {
    const timestamp = Date.now();
    return `${uuidv4()}-${timestamp}`;
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    loadSessions();
  }, [user]);

  const loadSessions = async () => {
    try {
      const sessionIds = await api.getAllSessionIds(user);
      const sessionsData = await Promise.all(
        sessionIds.map(async (id) => {
          const conversation = await api.getChatDetails(id);
          return { id, conversation };
        })
      );
      
      sessionsData.sort((a, b) => {
        const timeA = a.conversation?.[0]?.timestamp || '';
        const timeB = b.conversation?.[0]?.timestamp || '';
        return new Date(timeB) - new Date(timeA);
      });

      setSessions(sessionsData);
      
      if (!currentSessionId && sessionsData.length > 0) {
        selectSession(sessionsData[0].id);
      } else if (!currentSessionId) {
        handleNewSession();
      }
    } catch (error) {
      console.error('Failed to load sessions:', error);
      handleNewSession();
    }
  };

  const selectSession = async (sessionId) => {
    setCurrentSessionId(sessionId);
    try {
      const conversation = await api.getChatDetails(sessionId);
      const formattedMessages = [];
      conversation.forEach(turn => {
        formattedMessages.push({ content: turn.query, isUser: true });
        formattedMessages.push({ content: turn.response, isUser: false });
      });
      setMessages(formattedMessages);
    } catch (error) {
      console.error('Failed to load session:', error);
      setMessages([]);
    }
  };

  const handleNewSession = () => {
    const newSessionId = generateSessionId();
    setCurrentSessionId(newSessionId);
    setMessages([]);
  };

  const handlePromptSelect = (prompt) => {
    setSelectedPrompt(prompt);
    setShowPromptPreview(true);
    setShowPromptLibrary(false);
  };

  const handlePromptSend = (promptText) => {
    setShowPromptPreview(false);
    setSelectedPrompt(null);
    handleSendMessage(promptText);
  };

  const handleSendMessage = async (query) => {
    if (!currentSessionId || isLoading) return;

    const userMessage = { content: query, isUser: true };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setToolCallStatus('');

    let assistantMessage = '';
    const tempAssistantMessage = { content: '', isUser: false, isStreaming: true };
    setMessages(prev => [...prev, tempAssistantMessage]);

    try {
      await api.streamChat(
        user,
        currentSessionId,
        query,
        (chunk) => {
          assistantMessage += chunk;
          setMessages(prev => {
            const newMessages = [...prev];
            newMessages[newMessages.length - 1] = {
              content: assistantMessage,
              isUser: false,
              isStreaming: true
            };
            return newMessages;
          });
        },
        (toolCall) => {
          setToolCallStatus(toolCall);
        }
      );

      setMessages(prev => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1] = {
          content: assistantMessage,
          isUser: false,
          isStreaming: false
        };
        return newMessages;
      });

      const sessionExists = sessions.find(s => s.id === currentSessionId);
      if (!sessionExists) {
        await loadSessions();
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      setMessages(prev => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1] = {
          content: 'Sorry, an error occurred. Please try again.',
          isUser: false,
          isError: true
        };
        return newMessages;
      });
    } finally {
      setIsLoading(false);
      setToolCallStatus('');
    }
  };

  if (showPromptLibrary) {
    return (
      <div className="h-screen flex flex-col">
        <div className="flex-1">
          <PromptLibrary 
            user={user}
            onSelectPrompt={handlePromptSelect}
          />
        </div>
        <div className="border-t-2 border-black bg-white p-4">
          <button
            onClick={() => setShowPromptLibrary(false)}
            className="px-4 py-2 border-2 border-black hover:bg-black hover:text-white transition-colors"
          >
            Back to Chat
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-white">
      <Sidebar
        sessions={sessions}
        currentSessionId={currentSessionId}
        onSessionSelect={selectSession}
        onNewSession={handleNewSession}
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
      />

      <div className="flex-1 flex flex-col">
        <div className="border-b-2 border-black bg-white p-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-bold text-gray-900">Chat Assistant</h1>
            </div>
            <button
              onClick={() => setShowPromptLibrary(true)}
              className="flex items-center gap-2 px-4 py-2 border-2 border-black hover:bg-black hover:text-white transition-colors"
            >
              <BookOpen className="w-4 h-4" />
              Prompt Library
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center p-8">
              <div className="text-center max-w-md">
                <div className="w-16 h-16 mx-auto mb-4 border-2 border-black bg-red-600 flex items-center justify-center">
                  <span className="text-white text-2xl font-bold">AI</span>
                </div>
                <h2 className="text-xl font-semibold text-gray-900 mb-2">{getTimeBasedGreeting()}, {user}!</h2>
                <p className="text-gray-600">How can I help you today?</p>
              </div>
            </div>
          ) : (
            <div>
              {messages.map((message, index) => (
                <Message key={index} message={message} isUser={message.isUser} />
              ))}
              <AnimatePresence>
                {isLoading && <LoadingIndicator />}
              </AnimatePresence>
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <PromptSuggestions 
          onSelectPrompt={handlePromptSelect}
          onShowMore={() => setShowPromptLibrary(true)}
        />

        <ChatInput onSend={handleSendMessage} disabled={isLoading} />
      </div>

      {showPromptPreview && selectedPrompt && (
        <PromptPreview
          prompt={selectedPrompt}
          onSend={handlePromptSend}
          onCancel={() => {
            setShowPromptPreview(false);
            setSelectedPrompt(null);
          }}
        />
      )}
    </div>
  );
};

export default Chat;
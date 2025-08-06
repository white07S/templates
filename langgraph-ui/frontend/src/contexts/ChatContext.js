import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';
import { useAuth } from './AuthContext';

const ChatContext = createContext();

const API_BASE_URL = 'http://localhost:8000';

export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};

export const ChatProvider = ({ children }) => {
  const { user } = useAuth();
  const [chatHistory, setChatHistory] = useState([]);
  const [currentChat, setCurrentChat] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);

  // Load chat history when user logs in
  useEffect(() => {
    if (user?.userId) {
      loadChatHistory();
    }
  }, [user]);

  const loadChatHistory = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/history/${user.userId}`);
      setChatHistory(response.data.chats);
    } catch (error) {
      console.error('Failed to load chat history:', error);
    }
  };

  const createNewChat = () => {
    const sessionId = generateSessionId();
    setCurrentChat({ 
      sessionId, 
      messages: [],
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    });
    setMessages([]);
  };

  const selectChat = (chat) => {
    setCurrentChat(chat);
    
    // Transform chat messages to include proper structure for display
    const transformedMessages = [];
    if (chat.messages) {
      chat.messages.forEach(msg => {
        // Add user message
        if (msg.message) {
          transformedMessages.push({
            message: msg.message,
            response: '',
            timestamp: msg.timestamp,
            isUser: true
          });
        }
        
        // Add bot response
        if (msg.response) {
          transformedMessages.push({
            message: '',
            response: msg.response,
            timestamp: msg.timestamp,
            isUser: false,
            tools_used: msg.tools_used || []
          });
        }
      });
    }
    
    setMessages(transformedMessages);
  };

  const sendMessage = async (messageText) => {
    if (!messageText.trim() || isStreaming) return;

    const userMessage = {
      message: messageText,
      response: '',
      timestamp: new Date().toISOString(),
      isUser: true
    };

    const botMessage = {
      message: '',
      response: '',
      timestamp: new Date().toISOString(),
      isUser: false,
      tools_used: []
    };

    // Create new chat if none exists
    if (!currentChat) {
      createNewChat();
    }

    setMessages(prev => [...prev, userMessage, botMessage]);
    setIsStreaming(true);

    try {
      const sessionId = currentChat?.sessionId || generateSessionId();
      
      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: messageText,
          session_id: sessionId,
          user_id: user.userId,
          stream: true
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let fullResponse = '';
      let toolsUsed = [];
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        // Decode the chunk and add to buffer
        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        // Process complete lines from buffer
        const lines = buffer.split('\n');
        
        // Keep the last incomplete line in buffer
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim() === '') continue; // Skip empty lines
          
          if (line.startsWith('data: ')) {
            try {
              const jsonData = line.slice(6).trim();
              if (jsonData === '') continue; // Skip empty data lines
              
              const data = JSON.parse(jsonData);
              
              if (data.type === 'content') {
                fullResponse += data.data;
                // Update UI immediately with each character/word
                setMessages(prev => {
                  const newMessages = [...prev];
                  const lastMessage = newMessages[newMessages.length - 1];
                  if (lastMessage && !lastMessage.isUser) {
                    lastMessage.response = fullResponse;
                    lastMessage.timestamp = new Date().toISOString();
                  }
                  return newMessages;
                });
              } else if (data.type === 'tool_start') {
                console.log('Tool started:', data.tool);
              } else if (data.type === 'tool_end') {
                console.log('Tool completed:', data.output);
              } else if (data.type === 'done') {
                toolsUsed = data.tools_used || [];
                setMessages(prev => {
                  const newMessages = [...prev];
                  const lastMessage = newMessages[newMessages.length - 1];
                  if (lastMessage && !lastMessage.isUser) {
                    lastMessage.tools_used = toolsUsed;
                  }
                  return newMessages;
                });
              } else if (data.type === 'error') {
                console.error('Streaming error:', data.error);
                setMessages(prev => {
                  const newMessages = [...prev];
                  const lastMessage = newMessages[newMessages.length - 1];
                  if (lastMessage && !lastMessage.isUser) {
                    lastMessage.response = `Error: ${data.error}`;
                  }
                  return newMessages;
                });
                break; // Exit on error
              }
            } catch (e) {
              console.debug('JSON parse error (incomplete chunk):', e.message, 'Line:', line);
            }
          }
        }
      }

      // Update current chat and reload history
      if (!currentChat) {
        setCurrentChat({
          sessionId,
          messages: [userMessage, { ...botMessage, response: fullResponse, tools_used: toolsUsed }],
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        });
      }

      await loadChatHistory();

    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => {
        const newMessages = [...prev];
        const lastMessage = newMessages[newMessages.length - 1];
        if (lastMessage && !lastMessage.isUser) {
          lastMessage.response = 'Sorry, there was an error processing your message.';
        }
        return newMessages;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  const generateSessionId = () => {
    return 'session-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
  };

  const value = {
    chatHistory,
    currentChat,
    messages,
    isLoading,
    isStreaming,
    sendMessage,
    createNewChat,
    selectChat,
    loadChatHistory
  };

  return (
    <ChatContext.Provider value={value}>
      {children}
    </ChatContext.Provider>
  );
};
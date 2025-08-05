import React from 'react';
import { Thread, useAssistant } from '@assistant-ui/react';
import '@assistant-ui/react/dist/styles/index.css';

const API_BASE_URL = 'http://localhost:8080';

function App() {
  const assistant = useAssistant({
    api: {
      async chat({ messages, onUpdate }) {
        const lastMessage = messages[messages.length - 1];
        
        const response = await fetch(`${API_BASE_URL}/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: lastMessage.content,
            stream: true
          }),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullContent = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.type === 'content') {
                  fullContent += data.data;
                  onUpdate({
                    content: fullContent,
                    status: 'streaming'
                  });
                } else if (data.type === 'done') {
                  onUpdate({
                    content: fullContent,
                    status: 'complete'
                  });
                }
              } catch (e) {
                console.error('Parse error:', e);
              }
            }
          }
        }
      }
    }
  });

  return (
    <div className="h-screen flex flex-col">
      <Thread 
        assistant={assistant}
        className="flex-1"
        welcome={{
          message: "Hello! I'm your AI assistant powered by LangGraph and vLLM.",
          suggestions: [
            "What's the weather in New York?",
            "Calculate the square root of 144",
            "Search for information about LangGraph"
          ]
        }}
      />
    </div>
  );
}

npx create-react-app chat-app
cd chat-app
npm install @assistant-ui/react @assistant-ui/react-hook-form tailwindcss

export default App;

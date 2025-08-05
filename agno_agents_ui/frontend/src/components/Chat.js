import React from 'react';
import {
  AssistantRuntimeProvider,
  ThreadPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
  useLocalRuntime
} from '@assistant-ui/react';
import { createBackendAdapter } from './ChatAdapter';
import './assistant-ui.css';

const ChatComponent = () => {
  const runtime = useLocalRuntime(createBackendAdapter());

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="flex flex-col h-screen bg-gray-50">
        <header className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <h1 className="text-2xl font-semibold text-gray-900 py-4">
              Chat Assistant
            </h1>
          </div>
        </header>

        <ThreadPrimitive.Root className="flex-1 flex flex-col overflow-hidden">
          <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto p-4">
            <ThreadPrimitive.Empty>
              <div className="text-center p-8">
                <h2 className="text-xl font-semibold mb-2">Welcome to Chat Assistant</h2>
                <p className="text-gray-600">How can I help you today?</p>
              </div>
            </ThreadPrimitive.Empty>

            <ThreadPrimitive.Messages
              components={{
                UserMessage: () => (
                  <MessagePrimitive.Root className="mb-4">
                    <MessagePrimitive.If user>
                      <div className="flex justify-end">
                        <div className="bg-blue-500 text-white rounded-lg p-3 max-w-md">
                          <MessagePrimitive.Content />
                        </div>
                      </div>
                    </MessagePrimitive.If>
                  </MessagePrimitive.Root>
                ),
                AssistantMessage: () => (
                  <MessagePrimitive.Root className="mb-4">
                    <MessagePrimitive.If assistant>
                      <div className="flex justify-start">
                        <div className="bg-gray-200 text-gray-800 rounded-lg p-3 max-w-md">
                          <MessagePrimitive.Content />
                        </div>
                      </div>
                    </MessagePrimitive.If>
                  </MessagePrimitive.Root>
                )
              }}
            />

            <ThreadPrimitive.ScrollToBottom className="fixed bottom-24 right-8 z-10">
              <button className="bg-white rounded-full p-2 shadow-lg hover:shadow-xl transition-shadow">
                ↓
              </button>
            </ThreadPrimitive.ScrollToBottom>
          </ThreadPrimitive.Viewport>

          <div className="border-t bg-white p-4">
            <ComposerPrimitive.Root className="flex gap-2">
              <ComposerPrimitive.Input
                placeholder="Type your message..."
                className="flex-1 p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <ComposerPrimitive.Send className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors disabled:opacity-50">
                Send
              </ComposerPrimitive.Send>
            </ComposerPrimitive.Root>
          </div>
        </ThreadPrimitive.Root>
      </div>
    </AssistantRuntimeProvider>
  );
};

export default ChatComponent;
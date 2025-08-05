import axios from 'axios';

export const createBackendAdapter = () => ({
  async *run({ messages, abortSignal }) {
    try {
      // Get the last user message
      const lastMessage = messages[messages.length - 1];
      const userMessage = lastMessage?.content[0]?.text || '';

      const response = await axios({
        method: 'POST',
        url: 'http://localhost:8000/chat',
        data: { 
          message: userMessage,
          system_prompt: "You are a helpful AI assistant. Provide clear and concise responses."
        },
        responseType: 'stream',
        signal: abortSignal,
        adapter: 'fetch'
      });

      const reader = response.data.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let accumulatedContent = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Decode the chunk and add to buffer
        buffer += decoder.decode(value, { stream: true });
        
        // Process complete SSE messages from buffer
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6); // Remove 'data: ' prefix
            
            try {
              const data = JSON.parse(dataStr);
              
              if (data.type === 'content' && data.content) {
                accumulatedContent += data.content;
                
                yield {
                  content: [{
                    type: 'text',
                    text: accumulatedContent
                  }]
                };
              }
              // Ignore 'start' and 'done' type messages
            } catch (e) {
              // Skip invalid JSON
              console.error('Failed to parse SSE data:', e);
            }
          }
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        throw error;
      }
      yield {
        content: [{
          type: 'text',
          text: `Error: ${error.message}`
        }]
      };
    }
  }
});
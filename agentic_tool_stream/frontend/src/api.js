const BASE_URL = 'http://localhost:8000';

export const api = {
  async streamChat(userId, sessionId, query, onChunk, onToolCall) {
    const response = await fetch(`${BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_id: userId,
        session_id: sessionId,
        query: query,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.includes('🔧 Tool called:')) {
          onToolCall && onToolCall(line);
        } else if (line.includes('✅ Tool completed:')) {
          onToolCall && onToolCall(line);
        } else if (line.trim()) {
          onChunk(line + '\n');
        }
      }
    }

    if (buffer.trim()) {
      if (buffer.includes('🔧') || buffer.includes('✅')) {
        onToolCall && onToolCall(buffer);
      } else {
        onChunk(buffer);
      }
    }
  },

  async getAllSessionIds(userId) {
    const response = await fetch(`${BASE_URL}/get-all-session-id`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_id: userId,
      }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data.session_ids || [];
  },

  async getChatDetails(sessionId) {
    const response = await fetch(`${BASE_URL}/get-chat-details`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: sessionId,
      }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data.conversation || [];
  },
};
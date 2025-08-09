const BASE_URL = 'http://localhost:8000';

export const api = {
  async streamChat(userId, sessionId, query, onChunk, onToolCall) {
  const response = await fetch(`${BASE_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream'
    },
    body: JSON.stringify({ user_id: userId, session_id: sessionId, query })
  });

  if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by \n\n
    const frames = buffer.split('\n\n');
    buffer = frames.pop() || '';

    for (const frame of frames) {
      if (!frame.startsWith('data:')) continue;
      const json = frame.slice(5).trim();
      if (!json) continue;

      const msg = JSON.parse(json);
      switch (msg.type) {
        case 'content':
          onChunk && onChunk(msg.delta);
          break;
        case 'tool_start':
          onToolCall && onToolCall(`🔧 Tool called: ${msg.name}`);
          break;
        case 'tool_end':
          onToolCall && onToolCall(`✅ Tool completed: ${msg.name}`);
          break;
        case 'done':
          // optionally handle completion
          break;
      }
    }
  }

  // handle any trailing partial frame if needed (usually empty)
  if (buffer.startsWith('data:')) {
    const msg = JSON.parse(buffer.slice(5).trim());
    if (msg.type === 'content') onChunk && onChunk(msg.delta);
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

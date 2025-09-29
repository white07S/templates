const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const docsApi = {
  // Fetch list of all documents
  async getDocsList() {
    const response = await fetch(`${API_BASE_URL}/api/docs`);
    if (!response.ok) {
      throw new Error(`Failed to fetch docs list: ${response.statusText}`);
    }
    return response.json();
  },

  // Fetch specific document content
  async getDocContent(docId) {
    const response = await fetch(`${API_BASE_URL}/api/docs/${docId}`);
    if (!response.ok) {
      if (response.status === 404) {
        throw new Error(`Document '${docId}' not found`);
      }
      throw new Error(`Failed to fetch document: ${response.statusText}`);
    }
    return response.json();
  },

  // Fetch raw MDX content
  async getDocRaw(docId) {
    const response = await fetch(`${API_BASE_URL}/api/docs/${docId}/raw`);
    if (!response.ok) {
      throw new Error(`Failed to fetch raw document: ${response.statusText}`);
    }
    return response.text();
  },

  // Search documents
  async searchDocs(query) {
    const params = new URLSearchParams({ q: query });
    const response = await fetch(`${API_BASE_URL}/api/search?${params}`);
    if (!response.ok) {
      throw new Error(`Search failed: ${response.statusText}`);
    }
    return response.json();
  }
};
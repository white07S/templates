const API_BASE_URL = 'http://localhost:8000/api/glossary';

class GlossaryAPI {
  // Get all terms with pagination
  static async getAllTerms(page = 1, limit = 10) {
    try {
      const response = await fetch(`${API_BASE_URL}/terms?page=${page}&limit=${limit}`);
      if (!response.ok) throw new Error('Failed to fetch terms');
      return await response.json();
    } catch (error) {
      console.error('Error fetching terms:', error);
      throw error;
    }
  }

  // Get a single term by ID
  static async getTerm(termId) {
    try {
      const response = await fetch(`${API_BASE_URL}/terms/${termId}`);
      if (!response.ok) throw new Error('Failed to fetch term');
      return await response.json();
    } catch (error) {
      console.error('Error fetching term:', error);
      throw error;
    }
  }

  // Create a new term
  static async createTerm(termData, userId) {
    try {
      const response = await fetch(`${API_BASE_URL}/terms`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...termData,
          user_id: userId
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to create term');
      }
      return await response.json();
    } catch (error) {
      console.error('Error creating term:', error);
      throw error;
    }
  }

  // Update a term
  static async updateTerm(termId, termData, userId) {
    try {
      const response = await fetch(`${API_BASE_URL}/terms/${termId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...termData,
          user_id: userId
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to update term');
      }
      return await response.json();
    } catch (error) {
      console.error('Error updating term:', error);
      throw error;
    }
  }

  // Delete a term
  static async deleteTerm(termId) {
    try {
      const response = await fetch(`${API_BASE_URL}/terms/${termId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to delete term');
      }
      return await response.json();
    } catch (error) {
      console.error('Error deleting term:', error);
      throw error;
    }
  }

  // Search terms with pagination
  static async searchTerms(query, page = 1, limit = 10) {
    try {
      const response = await fetch(`${API_BASE_URL}/search?q=${encodeURIComponent(query)}&page=${page}&limit=${limit}`);
      if (!response.ok) throw new Error('Failed to search terms');
      return await response.json();
    } catch (error) {
      console.error('Error searching terms:', error);
      throw error;
    }
  }

  // Get pending changes (for review)
  static async getPendingChanges() {
    try {
      const response = await fetch(`${API_BASE_URL}/pending-changes`);
      if (!response.ok) throw new Error('Failed to fetch pending changes');
      return await response.json();
    } catch (error) {
      console.error('Error fetching pending changes:', error);
      throw error;
    }
  }
}

export default GlossaryAPI;
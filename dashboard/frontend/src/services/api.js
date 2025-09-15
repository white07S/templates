const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class ApiService {
  async request(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // Dataset endpoints
  async getDatasetStats(datasetType) {
    return this.request(`/datasets/${datasetType}/stats`);
  }

  async getDatasetRecords(datasetType, params = {}) {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== null && value !== undefined && value !== '') {
        searchParams.append(key, value);
      }
    });
    
    const queryString = searchParams.toString();
    return this.request(`/datasets/${datasetType}${queryString ? `?${queryString}` : ''}`);
  }

  async getRecordDetail(datasetType, recordId) {
    return this.request(`/datasets/${datasetType}/${recordId}`);
  }

  async getTaxonomies(datasetType = null) {
    const endpoint = datasetType ? `/datasets/${datasetType}/taxonomies` : '/taxonomies';
    return this.request(endpoint);
  }

  // Search across all datasets
  async searchRecords(params = {}) {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== null && value !== undefined && value !== '') {
        searchParams.append(key, value);
      }
    });
    
    const queryString = searchParams.toString();
    return this.request(`/search${queryString ? `?${queryString}` : ''}`);
  }

  // Feedback endpoints
  async submitFeedback(feedback) {
    return this.request(`/api/feedback`, {
      method: 'POST',
      body: JSON.stringify(feedback),
    });
  }

  async getFeedback(recordId) {
    return this.request(`/api/feedback/${recordId}`);
  }

  // Health check
  async getHealth() {
    return this.request('/health');
  }
}

export const apiService = new ApiService();
export default apiService;
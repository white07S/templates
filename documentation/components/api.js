import axios from 'axios';

// Base API URL configuration
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Create axios instance with default configuration
const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding auth tokens if needed
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    // const token = localStorage.getItem('authToken');
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`;
    // }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for handling common errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle common error scenarios
    if (error.response?.status === 401) {
      // Handle unauthorized access
      console.error('Unauthorized access');
    } else if (error.response?.status === 404) {
      console.error('Resource not found');
    } else if (error.response?.status === 500) {
      console.error('Server error');
    }
    return Promise.reject(error);
  }
);

/**
 * Documentation API endpoints
 */
const documentationAPI = {
  /**
   * Get all pages for navigation/sidebar
   * @returns {Promise} Array of page objects with hierarchy
   */
  getPages: async () => {
    const response = await apiClient.get('/api/pages');
    return response.data;
  },

  /**
   * Get specific page content
   * @param {string} pageId - The ID of the page to fetch
   * @returns {Promise} Page content object with title and MDX content
   */
  getPageContent: async (pageId) => {
    const response = await apiClient.get(`/api/page/${pageId}`);
    return response.data;
  },

  /**
   * Get navigation data for a specific page (previous/next pages)
   * @param {string} pageId - The current page ID
   * @returns {Promise} Navigation object with previous, current, and next page info
   */
  getNavigation: async (pageId) => {
    const response = await apiClient.get(`/api/navigation/${pageId}`);
    return response.data;
  },
};

/**
 * Search API endpoints
 */
const searchAPI = {
  /**
   * Search documentation content
   * @param {string} query - Search query string
   * @param {string} category - Optional category filter ('all', 'heading', 'content', 'code')
   * @returns {Promise} Array of search results
   */
  search: async (query, category = 'all') => {
    const params = { q: query };
    if (category !== 'all') {
      params.category = category;
    }
    const response = await apiClient.get('/api/search', { params });
    return response.data;
  },
};

/**
 * Media/Asset API endpoints
 */
const mediaAPI = {
  /**
   * Get image URL
   * @param {string} imagePath - Path to the image
   * @returns {string} Full URL to the image
   */
  getImageUrl: (imagePath) => {
    // Remove 'images/' prefix if it exists since we'll add it in the API path
    const cleanPath = imagePath.replace(/^images\//, '');
    return imagePath.startsWith('http')
      ? imagePath
      : `${API_URL}/api/image/${cleanPath}`;
  },
};

/**
 * Utility functions
 */
const utils = {
  /**
   * Get the base API URL
   * @returns {string} The base API URL
   */
  getBaseUrl: () => API_URL,

  /**
   * Check if the API is reachable
   * @returns {Promise<boolean>} True if API is reachable, false otherwise
   */
  healthCheck: async () => {
    try {
      await apiClient.get('/health');
      return true;
    } catch (error) {
      console.error('API health check failed:', error);
      return false;
    }
  },
};

// Export all API modules
export {
  apiClient,
  documentationAPI,
  searchAPI,
  mediaAPI,
  utils,
};

// Default export for convenience
export default {
  documentation: documentationAPI,
  search: searchAPI,
  media: mediaAPI,
  utils,
  client: apiClient,
};
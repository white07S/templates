import axios from 'axios';
import { API_BASE_URL } from '../utils/constants';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

export const taskAPI = {
  getAvailableTasks: () => api.get('/available-tasks'),
  
  submitTask: async (taskData) => {
    const formData = new FormData();
    formData.append('username', taskData.username);
    formData.append('data_type', taskData.data_type);
    formData.append('tasks', taskData.tasks);
    formData.append('file', taskData.file);
    
    return axios.post(`${API_BASE_URL}/submit-task`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
  },
  
  getUserTasks: async (username) => {
    const formData = new FormData();
    formData.append('username', username);
    
    return axios.post(`${API_BASE_URL}/user-tasks`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
  },
  
  getTaskStatus: (taskId) => api.get(`/task-status/${taskId}`),
  
  getTaskResult: (taskId) => {
    return axios.get(`${API_BASE_URL}/task-result/${taskId}`, {
      responseType: 'blob'
    });
  }
};

export default api;
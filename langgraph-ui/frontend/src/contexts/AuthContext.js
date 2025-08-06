import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';

const AuthContext = createContext();

const API_BASE_URL = 'http://localhost:8000';

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const savedUser = localStorage.getItem('chatApp_user');
    if (savedUser) {
      try {
        setUser(JSON.parse(savedUser));
      } catch (error) {
        localStorage.removeItem('chatApp_user');
      }
    }
    setLoading(false);
  }, []);

  const login = async (name) => {
    try {
      setLoading(true);
      const response = await axios.post(`${API_BASE_URL}/auth`, { name });
      
      const userData = {
        userId: response.data.user_id,
        name: response.data.name,
        message: response.data.message
      };
      
      setUser(userData);
      localStorage.setItem('chatApp_user', JSON.stringify(userData));
      return userData;
    } catch (error) {
      console.error('Login error:', error);
      throw new Error(error.response?.data?.detail || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('chatApp_user');
  };

  const value = {
    user,
    login,
    logout,
    loading
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
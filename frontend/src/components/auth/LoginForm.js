import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { LogIn } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import Button from '../common/Button';
import Input from '../common/Input';
import ErrorMessage from '../common/ErrorMessage';
import './LoginForm.css';

const LoginForm = () => {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [formData, setFormData] = useState({
    username: '',
    secretCode: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (!formData.username || !formData.secretCode) {
      setError('Please fill in all fields');
      return;
    }

    setLoading(true);
    try {
      await login(formData.username, formData.secretCode);
      navigate('/ticket');
    } catch (err) {
      setError('Invalid username or secret code');
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      className="login-form-container"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <div className="login-form">
        <div className="login-form-header">
          <h2 className="text-title">Welcome Back</h2>
          <p className="text-body">Sign in to access the Task Ticket Platform</p>
        </div>

        {error && (
          <ErrorMessage 
            message={error} 
            onClose={() => setError('')}
            autoClose
          />
        )}

        <form onSubmit={handleSubmit} className="login-form-fields">
          <Input
            label="Username"
            name="username"
            type="text"
            value={formData.username}
            onChange={handleChange}
            placeholder="e.g., preetam.sharma"
            required
            autoComplete="username"
          />

          <Input
            label="Secret Code"
            name="secretCode"
            type="password"
            value={formData.secretCode}
            onChange={handleChange}
            placeholder="Enter your secret code"
            required
            autoComplete="current-password"
          />

          <Button
            type="submit"
            variant="primary"
            size="large"
            loading={loading}
            className="w-full"
            icon={<LogIn size={20} />}
          >
            Sign In
          </Button>
        </form>

        <div className="login-form-footer">
          <p className="text-small">
            Don't have access? Contact your administrator.
          </p>
        </div>
      </div>
    </motion.div>
  );
};

export default LoginForm;
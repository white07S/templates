import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FileText, BarChart3, MessageSquare, BookOpen, ArrowRight } from 'lucide-react';
import Button from '../../components/common/Button';
import './Home.css';

const Home = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: <FileText size={32} />,
      title: 'Task Ticket Management',
      description: 'Submit and track data processing tasks with real-time status updates',
      action: () => navigate('/ticket'),
      buttonText: 'Go to Tickets'
    },
    {
      icon: <BarChart3 size={32} />,
      title: 'Data Dashboard',
      description: 'Visualize and analyze your processed data with interactive charts',
      action: () => navigate('/data-dashboard'),
      buttonText: 'View Dashboard'
    },
    {
      icon: <MessageSquare size={32} />,
      title: 'AI Chat',
      description: 'Get insights and ask questions about your data using AI',
      action: () => navigate('/chat'),
      buttonText: 'Start Chat'
    },
    {
      icon: <BookOpen size={32} />,
      title: 'Prompt Library',
      description: 'Access and manage reusable prompts for data analysis',
      action: () => navigate('/prompt-lib'),
      buttonText: 'Browse Library'
    }
  ];

  return (
    <div className="home-page">
      <div className="container">
        <motion.section
          className="hero-section"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className="text-title">Data Processing Platform</h1>
          <p className="hero-description">
            Professional data processing and analytics platform for risk management and compliance
          </p>
          <Button
            variant="primary"
            size="large"
            onClick={() => navigate('/ticket')}
            icon={<ArrowRight size={20} />}
          >
            Get Started
          </Button>
        </motion.section>

        <section className="features-section">
          <h2 className="text-heading">Platform Features</h2>
          <div className="features-grid">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                className="feature-card"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
                whileHover={{ scale: 1.02 }}
              >
                <div className="feature-icon">{feature.icon}</div>
                <h3 className="feature-title">{feature.title}</h3>
                <p className="feature-description">{feature.description}</p>
                <Button
                  variant="secondary"
                  size="medium"
                  onClick={feature.action}
                  className="feature-button"
                >
                  {feature.buttonText}
                </Button>
              </motion.div>
            ))}
          </div>
        </section>

        <section className="stats-section">
          <div className="stats-grid">
            <div className="stat-item">
              <h3 className="stat-number">5</h3>
              <p className="stat-label">Data Types</p>
            </div>
            <div className="stat-item">
              <h3 className="stat-number">15+</h3>
              <p className="stat-label">Processing Tasks</p>
            </div>
            <div className="stat-item">
              <h3 className="stat-number">24/7</h3>
              <p className="stat-label">Processing</p>
            </div>
            <div className="stat-item">
              <h3 className="stat-number">Secure</h3>
              <p className="stat-label">Data Handling</p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default Home;
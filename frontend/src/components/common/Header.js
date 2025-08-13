import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { LogOut, User } from 'lucide-react';
import Button from './Button';
import './Header.css';

const Header = () => {
  const navigate = useNavigate();
  const { user, logout, isAuthenticated } = useAuth();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <header className="header">
      <div className="container">
        <div className="header-content">
          <div className="header-logo" onClick={() => navigate('/')}>
            <h1 className="header-title">Task Ticket Platform</h1>
          </div>
          
          {isAuthenticated && (
            <div className="header-right">
              <div className="header-user">
                <User size={20} />
                <span className="header-username">{user?.username}</span>
              </div>
              <Button
                variant="ghost"
                size="small"
                onClick={handleLogout}
                icon={<LogOut size={18} />}
              >
                Logout
              </Button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;
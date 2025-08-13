import React from 'react';
import { NavLink } from 'react-router-dom';
import { Home, FileText, BarChart3, MessageSquare, BookOpen } from 'lucide-react';
import './Navigation.css';

const Navigation = () => {
  const navItems = [
    { path: '/', label: 'Home', icon: <Home size={20} /> },
    { path: '/ticket', label: 'Ticket', icon: <FileText size={20} /> },
    { path: '/data-dashboard', label: 'Data Dashboard', icon: <BarChart3 size={20} /> },
    { path: '/chat', label: 'Chat', icon: <MessageSquare size={20} /> },
    { path: '/prompt-lib', label: 'Prompt Library', icon: <BookOpen size={20} /> }
  ];

  return (
    <nav className="navigation">
      <div className="container">
        <ul className="nav-list">
          {navItems.map((item) => (
            <li key={item.path}>
              <NavLink
                to={item.path}
                className={({ isActive }) =>
                  `nav-link ${isActive ? 'nav-link-active' : ''}`
                }
              >
                {item.icon}
                <span>{item.label}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  );
};

export default Navigation;
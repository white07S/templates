import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Docs from './pages/Docs';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/docs/*" element={<Docs />} />
        </Routes>
      </div>
    </Router>
  );
}

function HomePage() {
  return (
    <div className="home-page">
      <header className="home-header">
        <h1>Welcome to Our Documentation</h1>
        <p>Professional documentation with UBS-inspired design</p>
        <Link to="/docs" className="docs-link">
          View Documentation
        </Link>
      </header>
    </div>
  );
}

export default App;

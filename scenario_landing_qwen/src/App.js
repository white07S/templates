
// src/App.js
import React from 'react';
import Header from './components/Header';
import Hero from './components/Hero';
import StatsDashboard from './components/StatsDashboard';
import FeaturesShowcase from './components/FeaturesShowcase';
import Compliance from './components/Compliance';
import Governance from './components/Governance';
import Footer from './components/Footer';

function App() {
  return (
    <div className="min-h-screen bg-white">
      <Header />
      <main>
        <Hero />
        <StatsDashboard />
        <FeaturesShowcase />
        <Compliance />
        <Governance />
      </main>
      <Footer />
    </div>
  );
}

export default App;
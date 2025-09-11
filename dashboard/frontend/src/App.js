import React from 'react';
import { DashboardProvider } from './contexts/DashboardContext';
import Dashboard from './components/Dashboard';

function App() {
  return (
    <DashboardProvider>
      <Dashboard user="admin" />
    </DashboardProvider>
  );
}

export default App;

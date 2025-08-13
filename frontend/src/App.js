import React from 'react';
import Ticket from './pages/Ticket/Ticket';
import './App.css';

function App() {
  const user = 'testuser';

  return (
    <div className="app">
      <main className="main-content">
        <Ticket user={user} />
      </main>
    </div>
  );
}

export default App;